# Type stubs for stable_baselines3.common.evaluation

from typing import Any

from gymnasium import Env
from stable_baselines3.common.vec_env import VecEnv

def evaluate_policy(
    model: Any,
    env: Env[Any, Any] | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Any = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float]: ...
