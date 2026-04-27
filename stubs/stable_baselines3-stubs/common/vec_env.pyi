# Type stubs for stable_baselines3.common.vec_env
# Covers only the symbols used by this project.

from typing import Any

import numpy as np
from gymnasium import Env

class VecEnv:
    num_envs: int

    def reset(self) -> np.ndarray: ...
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...
    def render(self, mode: str = "human") -> None: ...
    def close(self) -> None: ...
    def seed(self, seed: int | None = None) -> list[int | None]: ...

class DummyVecEnv(VecEnv):
    envs: list[Env[Any, Any]]

class VecEnvWrapper(VecEnv):
    venv: VecEnv

    def __init__(self, venv: VecEnv, **kwargs: Any) -> None: ...

class RunningMeanStd:
    mean: np.ndarray
    var: np.ndarray
    count: float

class VecNormalize(VecEnvWrapper):
    training: bool
    norm_reward: bool
    obs_rms: RunningMeanStd

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        **kwargs: Any,
    ) -> None: ...
    @staticmethod
    def load(load_path: str, venv: VecEnv) -> VecNormalize: ...
    def save(self, save_path: str) -> None: ...
    def normalize_obs(self, obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]: ...
