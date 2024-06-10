import torch
import torch.nn as nn
import numpy as np
from PPO import reparameterize, evaluate_lop_pi


class BCActor(nn.Module):

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
    

class BC:

    def __init__(self, buffer_exp, state_shape, action_shape, device=torch.device('cuda'), seed=0, batch_size=128, lr=1e-3):

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # デモンストレーションデータの保持するバッファ．
        self.buffer_exp = buffer_exp
        
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(self.actor, step_size=batch_size, gamma=0.9)

        # ネットワーク．
        self.actor = BCActor(state_shape, action_shape).to(device)
        self.optim = torch.optim.Adam(self.actor.parameters(), lr, weight_decay=1e-4)

        # その他パラメータ．
        self.batch_size = batch_size
        self.device = device

    def select_action(self, state):
        with torch.no_grad():
            action = self.actor(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0))
            print(f"Action: {action.cpu().numpy()[0]}")  # デバッグ用の出力を追加
        return action[0].cpu().numpy()

    def update(self):
        states, actions, _, _, _ = self.buffer_exp.sample(self.batch_size)
        loss = (self.actor(states) - actions).pow_(2).mean()

        self.optim.zero_grad()
        loss.backward(retain_graph=False)
        self.optim.step()    

class BCAgent:
    def __init__(self, policy):
        self.policy = policy

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float, device=self.policy.device)
        state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state)
        return action.cpu().numpy()[0]

    def run_episode(self, env, max_steps=1000):
        state = env.reset()
        done = False
        total_reward = 0
        timestep = 0

        while not done and timestep < max_steps:
            action = self.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            timestep += 1

        return total_reward

