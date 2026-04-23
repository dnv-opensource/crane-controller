# Type stubs for stable_baselines3.common.env_util

from collections.abc import Callable
from typing import Any

from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv

def make_vec_env(
    env_id: str | type[Env[Any, Any]] | Callable[..., Env[Any, Any]],
    n_envs: int = 1,
    seed: int | None = None,
    start_index: int = 0,
    monitor_dir: str | None = None,
    wrapper_class: type[Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[VecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
) -> VecEnv: ...
