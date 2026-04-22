import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env: gym.Env[object, object], disc_to_cont: list[np.ndarray]) -> None:
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, action: int) -> np.ndarray:
        return self.disc_to_cont[action]
