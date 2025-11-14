import time
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from game_2048 import Game_2048, Direction
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class game_2048_env(gym.Env):
    def __init__(self, rows=4, cols=4, render=False):
        super(game_2048_env, self).__init__()
        self.rows = rows
        self.cols = cols
        self.render_enabled = render
        self.game = Game_2048(rows, cols, graphics=render)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=12,
            shape=(1, rows, cols),
            dtype=np.float32,
        )
        self.last_obs = None

    def _to_obs(self, obs):
        return np.array(obs, dtype=np.float32).reshape(1, self.rows, self.cols)

    def reset(self, seed=None, options=None):
        self.highest_tile = 0
        self.score = 0
        super().reset(seed=seed, options=options)
        obs = self.game.reset()
        obs = self._to_obs(obs)
        self.last_obs = obs
        info = {}
        return obs, info

    def step(self, action):
        direction = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ][action]

        alive, obs, new_merged_tiles = self.game.next_state(direction)
        obs = self._to_obs(obs)
        reward = 0

        highest_tile = np.max(obs)
        for tile in new_merged_tiles:
            if tile == 11:
                reward += 100
            elif tile == 10:
                reward += 50
            elif tile == 9:
                reward += 20
            reward += tile
        empty_cells = np.sum(obs == 0)
        reward += empty_cells * 0.1
        
        if action == 2 or action == 1:
            reward *= 1.5
        # if left down tile is max value, give more reward
        if obs[0, self.rows-1, 0] == highest_tile:
            reward *= 1.5
        
        if obs.all() == self.last_obs.all():
            reward -= 1
        self.last_obs = obs
        reward /= 10.0
            
        # if highest_tile > self.highest_tile:
        #     reward = 2**highest_tile
        #     self.highest_tile = highest_tile
        # elif score > self.score:
        #     self.score = score
        #     reward = 1
        # else:
        #     reward = -1

        terminated = not alive
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_enabled:
            self.game.graphics.update_tiles(self.game.get_state())
            
class StepLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super(StepLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

class DuelingQNetwork(nn.Module):
    def __init__(self, features_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.fc = nn.Linear(features_dim, 128)
        # Dueling heads
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.value(x)
        adv = self.advantage(x)
        # combine value & advantage
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q
    
class DuelingDQNPolicy(DQNPolicy):
    def _build_q_net(self):
        features_dim = self.features_extractor.features_dim
        action_dim = self.action_space.n
        self.q_net = DuelingQNetwork(features_dim, action_dim)

class MyCNN_MLP_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=128):
        super(MyCNN_MLP_Extractor, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=3, padding=1)
        # compute output size automatically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            sample = F.relu(self.conv1(sample))
            sample = F.relu(self.conv2(sample))
            sample = F.relu(self.conv3(sample))
            n_flatten = sample.numel()

        self.fc1 = nn.Linear(n_flatten, 128)
        self.fc2 = nn.Linear(128, features_dim)
        self._features_dim = features_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

policy_kwargs = dict(
    features_extractor_class=MyCNN_MLP_Extractor,
    features_extractor_kwargs=dict(features_dim=128),
)

class RewardLossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.losses = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.rewards.append(ep_info["r"])
        return True

def train_and_save_model():
    env = game_2048_env(4, 4)
    env = StepLimitWrapper(env, max_steps=1000)
    def make_env(seed):
        def _init():
            env = StepLimitWrapper(game_2048_env(4,4), max_steps=1000)
            env.reset(seed=seed)
            env = Monitor(env)
            return env
        return _init
    vec_env = DummyVecEnv([make_env(i) for i in range(8)])

    model = DQN(
        DuelingDQNPolicy,
        vec_env,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        exploration_final_eps=0.005,
        exploration_fraction=0.998,
        exploration_initial_eps=0.9,
        buffer_size=500000,
        batch_size=128,
        gamma=0.9,
        learning_starts=1000,
        target_update_interval=200
    )
    callback = RewardLossCallback()
    model.learn(total_timesteps=1_500_000, log_interval=4, callback=callback)
    model.save("dqn_2048_model")
    
    def moving_average(x, window=20):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    smoothed_rewards = moving_average(callback.rewards, window=20)
    plt.figure(figsize=(7,4))
    plt.plot(smoothed_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Average Reward per Episode")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train_and_save_model()


