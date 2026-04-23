import logging

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class ClipReward(gym.RewardWrapper[object, object]):
    def __init__(self, env: gym.Env[object, object], min_reward: float, max_reward: float) -> None:
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: float) -> np.ndarray:
        _reward = np.array([reward], float)
        return np.clip(_reward, self.min_reward, self.max_reward)
