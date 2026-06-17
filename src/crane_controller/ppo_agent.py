"""PPO-based agent for the anti-pendulum environment."""

from __future__ import annotations

import dataclasses
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

from crane_controller.callbacks import EpRewardLogCallback
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable

plt.rcParams["figure.figsize"] = (10, 5)

logger = logging.getLogger(__name__)


_T_MIN_SETTLE_EPS = 0.05


def _t_min_settle(trace: list[float]) -> int:
    """Return the first step at which t_min permanently stays at/below the plateau.

    Scans from the end of *trace* to find the last step that was still above
    ``trace[-1] + _T_MIN_SETTLE_EPS``.  Returns that index + 2 (1-indexed step
    after the last unsettled step), or 1 if already settled from the start.
    """
    if not trace:
        return 0
    threshold = trace[-1] + _T_MIN_SETTLE_EPS
    for i in range(len(trace) - 1, -1, -1):
        if trace[i] > threshold:
            return i + 2
    return 1


@dataclasses.dataclass
class EpisodeResult:
    """Structured result returned by :meth:`ProximalPolicyOptimizationAgent.do_one_episode`."""

    start_speed: float
    ep_steps: int
    ep_reward: float
    terminated: bool
    truncated: bool
    no_crash: bool
    t_min_start: float
    t_min_min: float
    t_min_final: float
    t_min_mean_last100: float
    t_min_settle_step: int
    x_pos_final: float
    x_vel_final: float
    theta_final: float
    theta_dot_final: float
    energy_final: float
    acc_final: float


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
    seed : int or None, optional
        Random seed passed to PPO for reproducibility (default None).
    ent_coef : float, optional
        Entropy bonus coefficient (default 0.0).
    learning_rate : float, optional
        Adam learning rate (default 3e-4).
    clip_range : float, optional
        PPO clipping parameter (default 0.2). Lower = more conservative updates.
    n_steps : int, optional
        Timesteps collected per environment before each gradient update (default 2048).
        Increasing to 8192 gives ~11 complete episodes per update instead of ~3,
        producing more stable gradient estimates for long-horizon tasks.
    """

    def __init__(  # noqa: PLR0913
        self,
        env: Callable[..., AntiPendulumEnv],
        n_envs: int = 4,
        env_kwargs: dict[str, Any] | None = None,
        save_path: str | None = None,
        max_episode_steps: int = 1000,
        gamma: float = 0.99,
        seed: int | None = None,
        ent_coef: float = 0.0,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        n_steps: int = 2048,
    ) -> None:
        """Set up the agent for training. Use :meth:`load` for inference."""
        self.save_path = save_path
        self._max_episode_steps = max_episode_steps
        _mep = max_episode_steps
        raw_vec_env = make_vec_env(
            env_id=lambda **kw: TimeLimit(env(**kw), max_episode_steps=_mep),
            n_envs=n_envs,
            env_kwargs=env_kwargs,
        )
        self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True)
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            gamma=gamma,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            clip_range=clip_range,
            n_steps=n_steps,
            verbose=1 if n_envs == 1 else 0,
        )
        self.env: AntiPendulumEnv = self.vec_env.venv.envs[0]  # type: ignore[attr-defined]

    @classmethod
    def load(
        cls,
        env: Callable[..., AntiPendulumEnv],
        model_path: str | Path,
        env_kwargs: dict[str, Any] | None = None,
        max_episode_steps: int = 1000,
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

        Returns:
        -------
        ProximalPolicyOptimizationAgent
            Agent configured for inference with VecNormalize in evaluation mode.
        """
        instance = object.__new__(cls)
        instance._max_episode_steps = max_episode_steps  # noqa: SLF001
        _mep = max_episode_steps
        raw_vec_env = make_vec_env(
            env_id=lambda **kw: TimeLimit(env(**kw), max_episode_steps=_mep),
            n_envs=1,
            env_kwargs=env_kwargs,
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
        max_episode_steps: int = 1000,
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

        Returns:
        -------
        ProximalPolicyOptimizationAgent
            Agent configured for continued training with VecNormalize in training mode.
        """
        instance = object.__new__(cls)
        instance.save_path = save_path
        instance._max_episode_steps = max_episode_steps  # noqa: SLF001
        _mep = max_episode_steps
        raw_vec_env = make_vec_env(
            env_id=lambda **kw: TimeLimit(env(**kw), max_episode_steps=_mep),
            n_envs=n_envs,
            env_kwargs=env_kwargs,
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

        Returns:
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
        log_interval: int = 50_000,
        csv_path: str | None = None,
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
        log_interval : int, optional
            Timesteps between ep_rew_mean log lines printed alongside the
            progress bar (default 50 000). Ignored when progress_bar=False.
        csv_path : str or None, optional
            Path to write a CSV log file with per-interval metrics at the end
            of training (default None).
        """
        cb = (
            EpRewardLogCallback(
                total_timesteps,
                log_interval,
                csv_path=csv_path,
                max_episode_steps=self._max_episode_steps,
            )
            if progress_bar
            else None
        )
        _ = self.model.learn(
            total_timesteps,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
            callback=cb,
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

    def do_one_episode(
        self,
        seed: int = 1,
        save_png: str | None = None,
    ) -> EpisodeResult:
        """Run one episode on the non-vectorised, trained environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for the environment reset (default 1).
        save_png : str or None, optional
            If set, save a 7-panel trajectory plot to this path (default None).

        Returns:
        -------
        EpisodeResult
            Per-episode metrics including t_min stats, final crane state, and outcome.
        """
        obs, reset_info = self.env.reset(seed=seed)
        nan = float("nan")
        start_speed: float = self.env.unwrapped.initial_speed  # type: ignore[attr-defined]
        t_min_start_val: float = float(reset_info.get("t_min", nan))
        terminated = truncated = False
        ep_steps = 0
        ep_reward = 0.0
        t_min_trace: list[float] = []
        info: dict[str, float | int] = {}
        last_action: np.ndarray | int = 0
        while not terminated and not truncated:
            norm_obs = self.vec_env.normalize_obs(np.asarray(obs))
            action, _states = self.model.predict(np.asarray(norm_obs), deterministic=True)
            last_action = action
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_steps += 1
            ep_reward += float(reward)
            if "t_min" in info:
                t_min_trace.append(float(info["t_min"]))
        self.env.unwrapped.render(save_path=save_png)  # type: ignore[attr-defined, call-arg]
        env_u = self.env.unwrapped
        energy_final = 0.5 * float(env_u.wire.cm_v[0]) ** 2  # type: ignore[attr-defined]
        if env_u.continuous_actions:  # type: ignore[attr-defined]
            acc_final = float(np.asarray(last_action).flat[0]) * float(env_u.acc)  # type: ignore[attr-defined]
        else:
            acc_final = float(env_u.action_to_acc[int(last_action)])  # type: ignore[attr-defined]
        return EpisodeResult(
            start_speed=start_speed,
            ep_steps=ep_steps,
            ep_reward=ep_reward,
            terminated=terminated,
            truncated=truncated,
            no_crash=not info.get("crash", False),
            t_min_start=t_min_start_val,
            t_min_min=min(t_min_trace) if t_min_trace else nan,
            t_min_final=t_min_trace[-1] if t_min_trace else nan,
            t_min_mean_last100=float(np.mean(t_min_trace[-100:])) if t_min_trace else nan,
            t_min_settle_step=_t_min_settle(t_min_trace),
            x_pos_final=float(info.get("x_pos", nan)),
            x_vel_final=float(info.get("x_vel", nan)),
            theta_final=float(obs[2]),
            theta_dot_final=float(obs[3]),
            energy_final=energy_final,
            acc_final=acc_final,
        )
