import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class DiscreteActions(gym.ActionWrapper[object, int, np.ndarray]):  # type: ignore[type-arg]  # gymnasium wrapper generics are invariant
    def __init__(self, env: gym.Env[object, object], disc_to_cont: list[np.ndarray]) -> None:
        super().__init__(env)  # type: ignore[arg-type]  # gymnasium Env type params are invariant
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, action: int) -> np.ndarray:
        return self.disc_to_cont[action]
