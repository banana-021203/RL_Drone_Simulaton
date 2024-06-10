import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
import torch
import os
import csv

class Trainer:

    def __init__(self, env, algo, seed=0, num_steps=10**4, eval_interval=10**3, num_eval_episodes=5, max_episode_steps=3000, save_dir="weights"):

        # 評価用の環境．
        self.env = env
        self.env.seed(seed)

        # 学習アルゴリズム．
        self.algo = algo

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}

        # 学習ステップ数．
        self.num_steps = num_steps
        # 評価のインターバル．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes
        # 最大エピソード長
        self.max_episode_steps = max_episode_steps
        self.save_dir = save_dir

        self.csv_path = os.path.join(save_dir, "returns.csv")

        # CSVファイルのヘッダーを書き込む
        with open(self.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "return"])


    def train(self):
        # 学習開始の時間
        self.start_time = time()

        for step in range(1, self.num_steps + 1):
            self.algo.update()
            if step % self.eval_interval == 0:
                self.evaluate(step)

    def evaluate(self, step):
        #複数エピソード環境を動かし，平均収益を記録する
        total_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            done = False
            timestep = 0
            print(self.env.drone_position)
            print(self.env.target_position)

            while (not done):
                timestep += 1
                action = self.algo.select_action(state)
                #print("evaluate action:", action)
                state, reward, done, _ = self.env.step(action)
                total_return += reward

                if timestep == self.max_episode_steps:
                    done = True
                    print(_)
                    print(self.env.drone_position)
                    print(self.env.target_position)

        mean_return = total_return / self.num_eval_episodes
        self.returns['step'].append(step)
        self.returns['return'].append(mean_return)

         # 平均収益をCSVファイルに追記
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([step, mean_return])

        print(f'Num steps: {step:<6}   '
                f'Return: {mean_return:<5.1f}   '
                f'Time: {self.time}')
        print(self.env.drone_position)
        print(self.env.target_position)
        self.env.render()

    def visualize(self, max_episode_steps=3000):
        # 1エピソード分の軌跡を表示する
        env = self.env
        state = env.reset()
        done = False
        timestep = 0

        while (not done):
            timestep += 1

            action = self.algo.select_action(state)
            state, _, done, _ = env.step(action)

            if timestep == self.max_episode_steps:
                done = True

        env.render()
        #env.render2()
        del env


    def plot(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.tight_layout()

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))

    def save_weights(self, step):
        actor_path = os.path.join(self.save_dir, f"actor_{step}.pth")
        critic_path = os.path.join(self.save_dir, f"critic_{step}.pth")
        torch.save(self.algo.actor.state_dict(), actor_path)
        torch.save(self.algo.critic.state_dict(), critic_path)
        print(f"Saved weights at step {step}")

