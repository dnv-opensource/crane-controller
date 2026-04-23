import logging
from typing import SupportsFloat

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class ClipReward(gym.RewardWrapper[object, object]):
    def __init__(self, env: gym.Env[object, object], min_reward: float, max_reward: float) -> None:
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: SupportsFloat) -> np.ndarray:
        _reward = np.array([float(reward)], float)
        return np.clip(_reward, self.min_reward, self.max_reward)
