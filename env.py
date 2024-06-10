# 環境クラス
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random

class DroneEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float16)
        self.min_output =  0.1
        self.observation_space = spaces.Box(low=np.array([0, -np.pi, -np.pi]), high=np.array([10, np.pi, np.pi]), dtype=np.float16)

        self.drone_position_range = np.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
        self.target_position_range = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

        self.drone_diameter = 0.3

        # ゴール条件
        self.goal_distance = 0.5
        self.max_episode_steps = 6000

        self.viewer = None

        self.episode_drone_trajectories = []
        self.episode_target_trajectories = []

    def reset(self):
        self.viewer = None
        self.target_position = np.random.uniform(low=self.target_position_range[0], high=self.target_position_range[1])
        while True:
            self.drone_position = np.random.uniform(low=self.drone_position_range[0], high=self.drone_position_range[1])
            if np.linalg.norm(self.target_position - self.drone_position) >= 2.0:
                break

        self.drone_velocity = np.array([0, 0, 0], dtype=np.float16)

        self.drone_trajectory = []
        self.target_trajectory = []
        self.drone_trajectory.append(self.drone_position)
        self.target_trajectory.append(self.target_position)


        obs = self._get_obs()
        return obs

    def _get_obs(self):
        relative_position = self.target_position - self.drone_position
        distance = np.linalg.norm(relative_position)
        angles = np.arctan2(relative_position[1], relative_position[0]), np.arctan2(relative_position[2], np.sqrt(relative_position[0]**2 + relative_position[1]**2))

        return np.array([distance, *angles], dtype=np.float16)

    def step(self, action):
        # 行動から推力を計算
        thrust_z_1 = np.clip(action[0], self.min_output, 1) * 0.1  
        thrust_z_2 = np.clip(action[1], self.min_output, 1) * 0.1  
        thrust_z_3 = np.clip(action[2], self.min_output, 1) * 0.1  
        thrust_z_4 = np.clip(action[3], self.min_output, 1) * 0.1  

        # 揚力の計算
        lift_force = (thrust_z_1 + thrust_z_2 + thrust_z_3 + thrust_z_4) / (self.drone_diameter ** 3)

        # 推進力の計算
        thrust_x = (thrust_z_1 - thrust_z_2 + thrust_z_3 - thrust_z_4) / (self.drone_diameter ** 3)
        thrust_y = (thrust_z_1 + thrust_z_2 - thrust_z_3 - thrust_z_4) / (self.drone_diameter ** 3)

        acceleration_x = thrust_x
        acceleration_y = thrust_y
        acceleration_z = lift_force

        self.drone_velocity += np.array([acceleration_x, acceleration_y, acceleration_z]) * 0.1
        self.drone_position += self.drone_velocity * 0.1

        distance = np.linalg.norm(self.target_position - self.drone_position)

        reward = self.reward(distance)

        done = bool(distance < self.goal_distance)
        if done:
            print('-----GORL-----')
            reward += 10

        if np.any(self.drone_position < 0) or np.any(self.drone_position > 5):
            reward += -10

        self.drone_trajectory.append(self.drone_position)
        self.target_trajectory.append(self.target_position)

        if done:
            self.episode_drone_trajectories.append(np.array(self.drone_trajectory))
            self.episode_target_trajectories.append(np.array(self.target_trajectory))
            self.drone_trajectory = []
            self.target_trajectory = []

        obs = self._get_obs()

        return obs, reward, done, {}

    def reward(self, distance):
            relative_position = self.target_position - self.drone_position
            distance_reward = -distance
            angle_reward = -np.sum(np.abs(np.arctan2(relative_position[1], relative_position[0]))) - np.abs(np.arctan2(relative_position[2], np.sqrt(relative_position[0]**2 + relative_position[1]**2)))
            reward = 0.6*distance_reward + 0.4*angle_reward

            return reward

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = plt.figure(figsize=(10, 10))
            self.ax1 = self.viewer.add_subplot(221, projection='3d')
            self.ax2 = self.viewer.add_subplot(222)
            self.ax3 = self.viewer.add_subplot(223)
            self.ax4 = self.viewer.add_subplot(224)

            self.ax1.set_xlim3d(0, 5)
            self.ax1.set_ylim3d(0, 5)
            self.ax1.set_zlim3d(0, 5)
            self.ax1.set_xlabel('X')
            self.ax1.set_ylabel('Y')
            self.ax1.set_zlabel('Z')

            self.ax2.set_xlim(0, 5)
            self.ax2.set_ylim(0, 5)
            self.ax2.set_xlabel('X')
            self.ax2.set_ylabel('Y')
            self.ax2.set_title('Top View')

            self.ax3.set_xlim(0, 5)
            self.ax3.set_ylim(0, 5)
            self.ax3.set_xlabel('Y')
            self.ax3.set_ylabel('Z')
            self.ax3.set_title('Side View (Y-Z)')

            self.ax4.set_xlim(0, 5)
            self.ax4.set_ylim(0, 5)
            self.ax4.set_xlabel('X')
            self.ax4.set_ylabel('Z')
            self.ax4.set_title('Side View (X-Z)')

            self.drone_trajectory_plot, = self.ax1.plot([], [], [], 'r-')
            self.target_trajectory_plot, = self.ax1.plot([], [], [], 'g-')
            self.drone_plot, = self.ax1.plot([], [], [], 'ro', markersize=10)
            self.target_plot, = self.ax1.plot([], [], [], 'gx', markersize=10)

            self.drone_xy_trajectory_plot, = self.ax2.plot([], [], 'r-')
            self.target_xy_trajectory_plot, = self.ax2.plot([], [], 'g-')
            self.drone_xy_plot, = self.ax2.plot([], [], 'ro', markersize=5)
            self.target_xy_plot, = self.ax2.plot([], [], 'gx', markersize=5)

            self.drone_yz_trajectory_plot, = self.ax3.plot([], [], 'r-')
            self.target_yz_trajectory_plot, = self.ax3.plot([], [], 'g-')
            self.drone_yz_plot, = self.ax3.plot([], [], 'ro', markersize=5)
            self.target_yz_plot, = self.ax3.plot([], [], 'gx', markersize=5)

            self.drone_xz_trajectory_plot, = self.ax4.plot([], [], 'r-')
            self.target_xz_trajectory_plot, = self.ax4.plot([], [], 'g-')
            self.drone_xz_plot, = self.ax4.plot([], [], 'ro', markersize=5)
            self.target_xz_plot, = self.ax4.plot([], [], 'gx', markersize=5)

        # 各エピソードの軌跡を表示
        for i, (drone_traj, target_traj) in enumerate(zip(self.episode_drone_trajectories, self.episode_target_trajectories)):
            color = plt.cm.jet(i / len(self.episode_drone_trajectories))
            self.ax1.plot(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2], color=color, alpha=0.5)
            self.ax1.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], color=color, linestyle='--', alpha=0.5)

            self.ax2.plot(drone_traj[:, 0], drone_traj[:, 1], color=color, alpha=0.5)
            self.ax2.plot(target_traj[:, 0], target_traj[:, 1], color=color, linestyle='--', alpha=0.5)

            self.ax3.plot(drone_traj[:, 1], drone_traj[:, 2], color=color, alpha=0.5)
            self.ax3.plot(target_traj[:, 1], target_traj[:, 2], color=color, linestyle='--', alpha=0.5)

            self.ax4.plot(drone_traj[:, 0], drone_traj[:, 2], color=color, alpha=0.5)
            self.ax4.plot(target_traj[:, 0], target_traj[:, 2], color=color, linestyle='--', alpha=0.5)

        #現在のエピソードの軌跡を表示
        drone_trajectory = np.array(self.drone_trajectory)
        target_trajectory = np.array(self.target_trajectory)

        if len(drone_trajectory) > 0 and len(target_trajectory) > 0:
            self.ax1.plot(drone_trajectory[:, 0], drone_trajectory[:, 1], drone_trajectory[:, 2], 'r-')
            self.ax1.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2], 'g-')
            self.ax1.plot(drone_trajectory[-1, 0], drone_trajectory[-1, 1], drone_trajectory[-1, 2], 'ro', markersize=10)
            self.ax1.plot(target_trajectory[-1, 0], target_trajectory[-1, 1], target_trajectory[-1, 2], 'gx', markersize=10)

            self.ax2.plot(drone_trajectory[:, 0], drone_trajectory[:, 1], 'r-')
            self.ax2.plot(target_trajectory[:, 0], target_trajectory[:, 1], 'g-')
            self.ax2.plot(drone_trajectory[-1, 0], drone_trajectory[-1, 1], 'ro', markersize=5)
            self.ax2.plot(target_trajectory[-1, 0], target_trajectory[-1, 1], 'gx', markersize=5)

            self.ax3.plot(drone_trajectory[:, 1], drone_trajectory[:, 2], 'r-')
            self.ax3.plot(target_trajectory[:, 1], target_trajectory[:, 2], 'g-')
            self.ax3.plot(drone_trajectory[-1, 1], drone_trajectory[-1, 2], 'ro', markersize=5)
            self.ax3.plot(target_trajectory[-1, 1], target_trajectory[-1, 2], 'gx', markersize=5)

            self.ax4.plot(drone_trajectory[:, 0], drone_trajectory[:, 2], 'r-')
            self.ax4.plot(target_trajectory[:, 0], target_trajectory[:, 2], 'g-')
            self.ax4.plot(drone_trajectory[-1, 0], drone_trajectory[-1, 2], 'ro', markersize=5)
            self.ax4.plot(target_trajectory[-1, 0], target_trajectory[-1, 2], 'gx', markersize=5)


        # ドローンの向きを表示
        if len(drone_trajectory) > 0:
            drone_theta_z = np.arctan2(np.sqrt(self.drone_velocity[0]**2 + self.drone_velocity[1]**2), self.drone_velocity[2])
            drone_theta = np.arctan2(self.drone_velocity[0], self.drone_velocity[1])
            heading_x = drone_trajectory[-1, 0] + np.cos(drone_theta)
            heading_y = drone_trajectory[-1, 1] + np.sin(drone_theta)
            heading_z = drone_trajectory[-1, 2] + np.sin(drone_theta_z)
            self.ax1.plot([drone_trajectory[-1, 0], heading_x], [drone_trajectory[-1, 1], heading_y], [drone_trajectory[-1, 2], heading_z], 'r-')
            self.ax2.plot([drone_trajectory[-1, 0], heading_x], [drone_trajectory[-1, 1], heading_y], 'r-')
            self.ax3.plot([drone_trajectory[-1, 1], heading_y], [drone_trajectory[-1, 2], heading_z], 'r-')
            self.ax4.plot([drone_trajectory[-1, 0], heading_x], [drone_trajectory[-1, 2], heading_z], 'r-')

        plt.pause(1.0)
        plt.show()


    def close(self):
            if self.viewer is not None:
                plt.close(self.viewer)
                self.viewer = None



