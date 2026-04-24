"""PPO-based agent for the anti-pendulum environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

if TYPE_CHECKING:
    from collections.abc import Callable

    from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv

plt.rcParams["figure.figsize"] = (10, 5)

logger = logging.getLogger(__name__)


class ProximalPolicyOptimizationAgent:
    """Agent that learns a policy via the PPO algorithm.

    `PPO algorithm <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_.

    PPO agents can be saved as a zip file and re-loaded to avoid re-training.
    VecNormalize statistics are saved alongside the model as ``<name>_vecnorm.pkl``.

    Parameters
    ----------
    env : Callable[..., AntiPendulumEnv]
        Factory callable that creates the environment.
    n_envs : int, optional
        Number of parallel environments used during training.
        ``n_envs=0`` signals that a pre-trained agent should be loaded from
        file (default 4).
    env_kwargs : dict[str, Any] or None, optional
        Additional keyword arguments forwarded to the environment factory
        (default None).
    trained : tuple[str | Path, bool] or None, optional
        File name and save/load flag for the trained agent. Required when
        ``n_envs=0`` (default None).
    """

    def __init__(
        self,
        env: Callable[..., AntiPendulumEnv],
        n_envs: int = 4,
        env_kwargs: dict[str, Any] | None = None,
        trained: tuple[str | Path, bool] | None = None,
    ) -> None:
        """Initialize the PPO agent.

        See the class docstring for parameter descriptions.
        """
        self.trained = trained
        inference_only = n_envs <= 0
        _n_envs = 1 if inference_only else n_envs

        raw_vec_env = make_vec_env(env_id=env, n_envs=_n_envs, env_kwargs=env_kwargs)  # type: ignore[arg-type]

        if inference_only:
            assert self.trained is not None, "When no training is specified a saved model should be provided"
            stats_path = self._stats_path(str(self.trained[0]))
            if stats_path.exists():
                self.vec_env = VecNormalize.load(str(stats_path), raw_vec_env)
            else:
                self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.model = PPO.load(str(self.trained[0]), env=self.vec_env)
        else:
            self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True)
            if _n_envs == 1:
                self.model = PPO("MlpPolicy", self.vec_env, verbose=1)
            else:
                self.model = PPO("MlpPolicy", self.vec_env)
            self.trained = (
                trained[0] if trained is not None else f"ppo_{env.__name__}",  # type: ignore[attr-defined]
                False if trained is None else trained[1],
            )

        # Single unwrapped env for do_one_episode/evaluate without reconstructing a new crane.
        self.env = self.vec_env.venv.envs[0]  # type: ignore[attr-defined]

    @staticmethod
    def _stats_path(model_path: str) -> Path:
        """Return the path for the VecNormalize statistics file.

        Parameters
        ----------
        model_path : str
            Path to the model zip file.

        Returns
        -------
        Path
            Path to the ``<stem>_vecnorm.pkl`` statistics file alongside the model.
        """
        p = Path(model_path)
        return p.parent / f"{p.stem}_vecnorm.pkl"

    def do_training(self, total_timesteps: int = 25000, *, progress_bar: bool = True) -> None:
        """Train the PPO model.

        Parameters
        ----------
        total_timesteps : int, optional
            Number of training timesteps (default 25000).
        progress_bar : bool, optional
            Whether to display a progress bar during training (default True).
        """
        _ = self.model.learn(total_timesteps, progress_bar=progress_bar)
        if self.trained is not None and self.trained[1] and self.env.render_mode != "play-back":
            self.model.save(str(self.trained[0]))
            self.vec_env.save(str(self._stats_path(str(self.trained[0]))))

    def evaluate(self, n_episodes: int = 10) -> None:
        """Evaluate the trained policy and log results.

        Parameters
        ----------
        n_episodes : int, optional
            Number of evaluation episodes (default 10).
        """
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=n_episodes)
        self.vec_env.training = True
        self.vec_env.norm_reward = True
        logger.info("Mean:%s, stdev:%s", mean_reward, std_reward)

    def do_one_episode(self, seed: int = 1) -> None:
        """Run one episode on the non-vectorised, trained environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for the environment reset (default 1).
        """
        obs, _ = self.env.reset(seed=seed)
        terminated = truncated = False
        while not terminated and not truncated:
            norm_obs = self.vec_env.normalize_obs(obs)
            action, _states = self.model.predict(np.asarray(norm_obs), deterministic=True)
            obs, _rewards, terminated, truncated, _ = self.env.step(int(action))
        self.env.render()
