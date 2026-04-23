"""Gymnasium reward wrapper that clips rewards to a specified range."""

import logging
from typing import SupportsFloat

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class ClipReward(gym.RewardWrapper[object, object]):
    """Gymnasium reward wrapper that clips rewards to a fixed range."""

    def __init__(self, env: gym.Env[object, object], min_reward: float, max_reward: float) -> None:
        """Initialize the reward clipper.

        Parameters
        ----------
        env : gym.Env[object, object]
            The environment to wrap.
        min_reward : float
            Lower bound for clipped rewards.
        max_reward : float
            Upper bound for clipped rewards.
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: SupportsFloat) -> np.ndarray:
        """Clip the reward to the configured range.

        Parameters
        ----------
        reward : SupportsFloat
            Raw reward from the wrapped environment.

        Returns
        -------
        np.ndarray
            Single-element array with the clipped reward.
        """
        _reward = np.array([float(reward)], float)
        return np.clip(_reward, self.min_reward, self.max_reward)
