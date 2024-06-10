import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + noises * stds
    actions = torch.tanh(us)
    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def calculate_advantage(values, rewards, dones, gamma=0.995, lambd=0.997):


    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    gaes = deltas.clone()  # gaesを初期化する方法を変更

    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] += gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    targets = gaes + values[:-1]
    return targets, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):

        self.states = torch.empty((buffer_size + 1, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

        self._p = 0
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self._p = (self._p + 1) % self.buffer_size

    def append_last_state(self, last_state):
        assert self._p == 0, 'Buffer needs to be full before appending last_state.'
        self.states[self.buffer_size].copy_(torch.from_numpy(last_state))


    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis

    def sample(self, batch_size):
        indices = np.random.randint(0, self.buffer_size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.log_pis[indices]
        )

class PPOActor(nn.Module):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=state_shape[0], hidden_size=128, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_shape[0]),
            nn.Tanh()
        )

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0], device = device))

    def forward(self, states):

        # LSTMの初期化
        h0 = torch.zeros(1, states.size(0), self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(1, states.size(0), self.lstm.hidden_size, device=self.device)

        # LSTMの計算
        lstm_output, _ = self.lstm(states.unsqueeze(1), (h0, c0))
        lstm_output = lstm_output.squeeze(1)

        return self.net(lstm_output)

    def sample(self, states):
        return reparameterize(self.forward(states), self.log_stds)


    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.forward(states), self.log_stds, actions)


class PPOCritic(nn.Module):

    def __init__(self, state_shape, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=state_shape[0], hidden_size=64, batch_first=True, num_layers=1)


        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, states):
        states = states.unsqueeze(1)

        # バッチサイズを取得
        batch_size = states.size(0)

        # sequence_lengthを1に設定
        sequence_length = 1

        # LSTMの初期化
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)


        lstm_output, _ = self.lstm(states, (h0, c0))
        lstm_output = lstm_output.squeeze(1)

        return self.net(lstm_output)


class PPO:

    def __init__(self, state_shape, action_shape, actor = None, device=torch.device('cuda'), seed=0,
                batch_size=500, lr=1e-3, gamma=0.995,  rollout_length=500, epoch_ppo=50,
                clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10., env = None):

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.env = env

        # データ保存用のバッファ．
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        if actor is None:
            self.actor = PPOActor(state_shape, action_shape).to(device)
        else:
            self.actor = actor.to(device)

        self.critic = PPOCritic(state_shape).to(device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        # その他パラメータ．
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        # print(state)
        with torch.no_grad():
            action = self.actor(state)
            #print(f"Action: {action.cpu().numpy()[0]}")  # デバッグ用の出力を追加
        return action.cpu().numpy()[0]

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        # バッファにデータを追加する．
        self.buffer.append(state, action, reward, mask, log_pi)

        # 最大ステップ数に到達したことでエピソードが終了した場合は，終了シグナルをFalseにする．
        mask = False if t == self.env._max_episode_steps else done

        # バッファにデータを追加する．
        self.buffer.append(state, action, reward, mask, log_pi)

        # ロールアウトの終端に達したら，最終状態をバッファに追加する．
        if step % self.rollout_length == 0:
            self.buffer.append_last_state(next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        states, actions, rewards, dones, log_pis = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis)

    def update_ppo(self, states, actions, rewards, dones, log_pis):
        with torch.no_grad():
            values = self.critic(states)

        # エピソードごとのステップ数を計算
        episode_lengths = []
        current_length = 0
        for done in dones:
            current_length += 1
            if done:
                episode_lengths.append(current_length)
                current_length = 0

        # 全ての報酬をステップ数で割る
        discounted_rewards = []
        current_episode_length = 0
        for reward, done in zip(rewards, dones):
            if done:
                current_episode_length = 0
            current_episode_length += 1
            discounted_reward = reward / current_episode_length
            discounted_rewards.append(discounted_reward)
        discounted_rewards = torch.tensor(discounted_rewards, device=self.device, dtype=torch.float32)

        # GAEを計算する
        targets, advantages = calculate_advantage(values, discounted_rewards, dones, self.gamma, self.lambd)

        # PPOを更新する
        for _ in range(self.epoch_ppo):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()