"""PPO-based agent for the anti-pendulum environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
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

    PPO agents can be saved as a zip file, re-loaded via :meth:`load` for inference,
    or re-loaded via :meth:`resume` to continue training.
    VecNormalize statistics are saved alongside the model as ``<name>_vecnorm.pkl``.

    Parameters
    ----------
    env : Callable[..., AntiPendulumEnv]
        Factory callable that creates the environment.
    n_envs : int, optional
        Number of parallel environments used during training (default 4).
    env_kwargs : dict[str, Any] or None, optional
        Additional keyword arguments forwarded to the environment factory (default None).
    save_path : str or None, optional
        File path for saving the trained model and VecNormalize statistics.
        If None, the model is not saved after training (default None).
    max_episode_steps : int, optional
        Maximum steps per episode enforced via a TimeLimit wrapper (default 3000).
        Ensures episodes always end, even when a plateau agent never triggers the
        environment's own termination condition.
    gamma : float, optional
        Discount factor for future rewards (default 0.99). Higher values (e.g. 0.999)
        extend the effective planning horizon, which can improve policy quality on
        long episodes at the cost of slower value function convergence.
    """

    def __init__(
        self,
        env: Callable[..., AntiPendulumEnv],
        n_envs: int = 4,
        env_kwargs: dict[str, Any] | None = None,
        save_path: str | None = None,
        max_episode_steps: int = 3000,
        gamma: float = 0.99,
    ) -> None:
        """Set up the agent for training. Use :meth:`load` for inference."""
        self.save_path = save_path
        raw_vec_env = make_vec_env(
            env_id=env,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            wrapper_class=TimeLimit,  # type: ignore[arg-type]
            wrapper_kwargs={"max_episode_steps": max_episode_steps},
        )
        self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True)
        self.model = PPO("MlpPolicy", self.vec_env, gamma=gamma, verbose=1 if n_envs == 1 else 0)
        self.env: AntiPendulumEnv = self.vec_env.venv.envs[0]  # type: ignore[attr-defined]

    @classmethod
    def load(
        cls,
        env: Callable[..., AntiPendulumEnv],
        model_path: str | Path,
        env_kwargs: dict[str, Any] | None = None,
    ) -> ProximalPolicyOptimizationAgent:
        """Load a trained agent for inference.

        Parameters
        ----------
        env : Callable[..., AntiPendulumEnv]
            Factory callable that creates the environment.
        model_path : str or Path
            Path to the saved model zip file.
        env_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments forwarded to the environment factory (default None).

        Returns
        -------
        ProximalPolicyOptimizationAgent
            Agent configured for inference with VecNormalize in evaluation mode.
        """
        instance = object.__new__(cls)
        raw_vec_env = make_vec_env(
            env_id=env,
            n_envs=1,
            env_kwargs=env_kwargs,
            wrapper_class=TimeLimit,  # type: ignore[arg-type]
            wrapper_kwargs={"max_episode_steps": 3000},
        )
        stats_path = cls._stats_path(str(model_path))
        if stats_path.exists():
            instance.vec_env = VecNormalize.load(str(stats_path), raw_vec_env)
        else:
            instance.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False)
        instance.vec_env.training = False
        instance.vec_env.norm_reward = False
        instance.model = PPO.load(str(model_path), env=instance.vec_env)
        instance.env = instance.vec_env.venv.envs[0]  # type: ignore[attr-defined]
        instance.save_path = None
        return instance

    @classmethod
    def resume(
        cls,
        env: Callable[..., AntiPendulumEnv],
        model_path: str | Path,
        env_kwargs: dict[str, Any] | None = None,
        save_path: str | None = None,
        n_envs: int = 4,
        max_episode_steps: int = 3000,
    ) -> ProximalPolicyOptimizationAgent:
        """Load a saved agent to continue training.

        Parameters
        ----------
        env : Callable[..., AntiPendulumEnv]
            Factory callable that creates the environment.
        model_path : str or Path
            Path to the saved model zip file.
        env_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments forwarded to the environment factory (default None).
        save_path : str or None, optional
            File path for saving the model after further training (default None).
        n_envs : int, optional
            Number of parallel environments for continued training (default 4).
        max_episode_steps : int, optional
            Maximum steps per episode enforced via a TimeLimit wrapper (default 3000).

        Returns
        -------
        ProximalPolicyOptimizationAgent
            Agent configured for continued training with VecNormalize in training mode.
        """
        instance = object.__new__(cls)
        instance.save_path = save_path
        raw_vec_env = make_vec_env(
            env_id=env,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            wrapper_class=TimeLimit,  # type: ignore[arg-type]
            wrapper_kwargs={"max_episode_steps": max_episode_steps},
        )
        stats_path = cls._stats_path(str(model_path))
        if stats_path.exists():
            instance.vec_env = VecNormalize.load(str(stats_path), raw_vec_env)
            instance.vec_env.training = True
            instance.vec_env.norm_reward = True
        else:
            instance.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True)
        instance.model = PPO.load(str(model_path), env=instance.vec_env)
        instance.env = instance.vec_env.venv.envs[0]  # type: ignore[attr-defined]
        return instance

    def _save_reward_plot(self, save_path: str) -> None:
        """Save a scatter plot of training rewards to a PNG file alongside the model.

        Collects ``reward_stats`` from all vectorized environments and saves a
        scatter plot of episode rewards vs training step to ``<save_path>.png``.
        Does nothing if no episodes completed during training.

        Parameters
        ----------
        save_path : str
            Path to the saved model zip file. The plot is written to the same
            location with a ``.png`` extension.
        """
        reward_stats: list[list[float]] = []
        for env in self.vec_env.venv.envs:  # type: ignore[attr-defined]
            reward_stats.extend(env.unwrapped.reward_stats)  # type: ignore[attr-defined]
        if not reward_stats:
            logger.warning("No episode reward stats found; skipping reward plot")
            return
        steps = [r[0] for r in reward_stats]
        rewards = [r[1] for r in reward_stats]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(steps, rewards, s=10, alpha=0.6, color="steelblue")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Final reward")
        ax.set_title(f"Training rewards - {Path(save_path).stem}")
        fig.tight_layout()
        plot_path = str(Path(save_path).with_suffix(".png"))
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info("Reward plot saved to %s", plot_path)

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

    def do_training(
        self,
        total_timesteps: int = 25000,
        *,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train the PPO model.

        Parameters
        ----------
        total_timesteps : int, optional
            Number of training timesteps (default 25000).
        progress_bar : bool, optional
            Whether to display a progress bar during training (default True).
        reset_num_timesteps : bool, optional
            Whether to reset the internal timestep counter before training.
            Set to False when resuming to preserve the learning rate schedule
            (default True).
        """
        _ = self.model.learn(
            total_timesteps,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        if self.save_path is not None and self.env.render_mode != "play-back":
            self.model.save(self.save_path)
            self.vec_env.save(str(self._stats_path(self.save_path)))
            self._save_reward_plot(self.save_path)

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
            norm_obs = self.vec_env.normalize_obs(np.asarray(obs))
            action, _states = self.model.predict(np.asarray(norm_obs), deterministic=True)
            obs, _rewards, terminated, truncated, _ = self.env.step(int(action))
        self.env.render()
