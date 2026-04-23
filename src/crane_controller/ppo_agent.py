from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv

plt.rcParams["figure.figsize"] = (10, 5)

logger = logging.getLogger(__name__)


class ProximalPolicyOptimizationAgent:
    """Agent which learns a policy via PPO algorithm to solve the task at hand.

    `PPO algorithm <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_.

    PPO agents can be saved as zip file and re-loaded to avoid re-training.

    Args:
        env (gym.Env): the environment the agent is acting on.
        n_envs (int) = 4: The number of environments to used during training.
          n_envs=0 signals that the trained agent should be loaded from file.
        env_kwargs (dict): Optional possibility to provide additional kwargs for environment
        trained (str|Path): Optional file name for saving/loading the trained agent. Required for n_envs=0
    """

    def __init__(
        self,
        env: Callable[..., AntiPendulumEnv],
        n_envs: int = 4,
        env_kwargs: dict[str, Any] | None = None,
        trained: tuple[str | Path, bool] | None = None,
    ) -> None:
        self.trained = trained
        if env_kwargs is None:
            self.env = env()
        else:
            self.env = env(**env_kwargs)
        _n_envs = n_envs = 1 if n_envs <= 0 else n_envs
        self.vec_env = make_vec_env(env_id=env, n_envs=_n_envs, env_kwargs=env_kwargs)
        if n_envs <= 0:
            assert self.trained is not None, "When no training is specified a saved model should be provided"
            self.model = PPO.load(self.trained[0])
        elif n_envs == 1:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            if trained is not None:
                self.trained = (trained[0], trained[1])
        else:
            self.model = PPO("MlpPolicy", self.vec_env)
            self.trained = (
                trained[0] if trained is not None else f"ppo_{env.__name__}",
                False if trained is None else trained[1],
            )

    def do_training(self, total_timesteps: int = 25000, *, progress_bar: bool = True) -> None:
        _ = self.model.learn(total_timesteps, progress_bar=progress_bar)
        if self.trained is not None and self.trained[1] and self.env.render_mode != "play-back":
            self.model.save(self.trained[0])

    def evaluate(self, n_episodes: int = 10) -> None:
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_episodes)
        logger.info("Mean:%s, stdev:%s", mean_reward, std_reward)

    def do_one_episode(self, seed: int = 1) -> None:
        """Do one episode on the non-vectorized, trained environment."""
        obs, _ = self.env.reset(seed=seed)
        terminated = truncated = False
        while not terminated and not truncated:
            action, _states = self.model.predict(np.asarray(obs))
            obs, _rewards, terminated, truncated, _ = self.env.step(int(action))
        self.env.render()
