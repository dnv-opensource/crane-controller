import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[object, object]) -> None:
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["target"] - observation["agent"]
