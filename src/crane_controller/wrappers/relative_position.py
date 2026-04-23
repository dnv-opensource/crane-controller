import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class RelativePosition(gym.ObservationWrapper[dict[str, np.ndarray], object, np.ndarray]):
    def __init__(self, env: gym.Env[object, object]) -> None:
        super().__init__(env)  # type: ignore[arg-type]  # gymnasium Env type params are invariant
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)  # type: ignore[assignment]  # Box is compatible with Space

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:  # type: ignore[override]  # transforms dict obs to ndarray
        return observation["target"] - observation["agent"]
