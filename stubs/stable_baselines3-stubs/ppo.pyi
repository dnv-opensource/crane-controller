# Type stubs for stable_baselines3.ppo
# Covers only the symbols used by this project.

from io import BufferedIOBase
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import Env
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class PPO:
    def __init__(self, policy: str, env: Env[Any, Any] | VecEnv, *, verbose: int = 0, **kwargs: Any) -> None: ...
    @classmethod
    def load(
        cls,
        path: str | Path | BufferedIOBase,
        env: Env[Any, Any] | VecEnv | None = None,
        device: torch.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> PPO: ...
    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | list[BaseCallback] | None = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> PPO: ...
    def save(self, path: str | Path) -> None: ...
    def predict(
        self,
        observation: NDArray[np.floating[Any]],
        state: NDArray[np.floating[Any]] | None = None,
        episode_start: NDArray[np.bool_] | None = None,
        deterministic: bool = False,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]] | None]: ...
