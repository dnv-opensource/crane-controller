"""Experiment configuration dataclasses and serialisation utilities."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for the five reward contributions in AntiPendulumEnv.

    Parameters
    ----------
    energy : float
        Weight for the pendulum energy term (default 1.0).
    positional : float
        Weight for the away-from-origin penalty (default 0.0015).
    time : float
        Weight for the time penalty (default 0.0).
    position : float
        Weight for the distance-from-origin penalty ``-|x|`` (default 0.005).
    acceleration : float
        Weight for the actuation-effort penalty ``-|acc|`` (default 0.01).
    terminal_penalty : float
        One-time reward added on truncation (out-of-bounds crash).  Set to a
        negative value to penalise early episode termination; 0.0 disables it
        (default 0.0).
    goal_tube_reward : float
        Per-step reward added every step the full 4D state lies inside the
        goal set (|x| < GOAL_EPS_X, |x_dot| < GOAL_EPS_X_DOT,
        |theta - pi| < GOAL_EPS_THETA, |theta_dot| < GOAL_EPS_THETA_DOT).
        Positive values reward the policy for reaching and staying in the
        settled state; under discounting (gamma=0.99), earlier settling
        accumulates more reward than later settling, giving an implicit
        speed-of-settling preference without any additional machinery.
        0.0 disables the term (default 0.0 — no change to existing configs).
        Symmetric counterpart to terminal_penalty: terminal_penalty penalises
        crashing (a one-time event); goal_tube_reward rewards settling
        (a per-step flow during the dwell period and beyond).
    pbrs : bool
        If True, the energy, position, and crane_velocity reward terms are
        applied as potential-based reward shaping F(s,s') = γΦ(s') − Φ(s)
        rather than as raw per-step penalties. The set of optimal policies is
        preserved under this transformation (Ng, Harada & Russell 1999).
        The positional, time, acceleration, and other terms are unaffected.
        Default False — preserves existing behaviour for all current configs.
    angle : float
        Weight for the squared pendulum angle penalty ``-theta^2`` (default 0.0).
    angular_velocity : float
        Weight for the squared angular velocity penalty ``-theta_dot^2`` (default 0.0).
        Uses pure angular velocity ``(cm_v[0] - origin_v[0]) / wire.length``,
        excluding crane translation.
    crane_velocity : float
        Weight for the squared crane velocity term ``+x_dot^2`` (default 0.0).
        Positive values reward crane velocity; use a negative value to penalise it.
    crane_acceleration : float
        Weight for the squared crane acceleration penalty ``-x_ddot^2`` (default 0.0).
        Equals the control action squared (acc = action * self.acc).
    angular_acceleration : float
        Weight for the squared angular acceleration penalty ``-theta_ddot^2`` (default 0.0).
        Computed as a one-step finite difference of theta_dot; zero on the first step
        after each reset.
    t_min_crane : float
        Weight for the minimum-time-to-origin penalty ``-t_min`` (default 0.0).
        ``t_min`` is the optimal bang-bang time for the crane to reach ``x=0`` at rest
        given current position and velocity.  Captures both crane position and velocity
        in a single physically grounded signal.
    """

    energy: float = 1.0
    positional: float = 0.0015
    time: float = 0.0
    position: float = 0.005
    acceleration: float = 0.01
    terminal_penalty: float = 0.0
    goal_tube_reward: float = 0.0
    pbrs: bool = False
    angle: float = 0.0
    angular_velocity: float = 0.0
    crane_velocity: float = 0.0
    crane_acceleration: float = 0.0
    angular_acceleration: float = 0.0
    t_min_crane: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> RewardConfig:
        """Instantiate from a mapping, filling missing keys with defaults.

        Parameters
        ----------
        d : dict[str, object]
            Mapping of field names to weight values. Unknown keys are ignored.

        Returns:
        -------
        RewardConfig
            Populated instance.
        """
        defaults = cls()
        return cls(
            energy=float(d.get("energy", defaults.energy)),  # type: ignore[arg-type]
            positional=float(d.get("positional", defaults.positional)),  # type: ignore[arg-type]
            time=float(d.get("time", defaults.time)),  # type: ignore[arg-type]
            position=float(d.get("position", defaults.position)),  # type: ignore[arg-type]
            acceleration=float(d.get("acceleration", defaults.acceleration)),  # type: ignore[arg-type]
            terminal_penalty=float(d.get("terminal_penalty", defaults.terminal_penalty)),  # type: ignore[arg-type]
            goal_tube_reward=float(d.get("goal_tube_reward", defaults.goal_tube_reward)),  # type: ignore[arg-type]
            pbrs=bool(d.get("pbrs", defaults.pbrs)),
            angle=float(d.get("angle", defaults.angle)),  # type: ignore[arg-type]
            angular_velocity=float(d.get("angular_velocity", defaults.angular_velocity)),  # type: ignore[arg-type]
            crane_velocity=float(d.get("crane_velocity", defaults.crane_velocity)),  # type: ignore[arg-type]
            crane_acceleration=float(d.get("crane_acceleration", defaults.crane_acceleration)),  # type: ignore[arg-type]
            angular_acceleration=float(d.get("angular_acceleration", defaults.angular_acceleration)),  # type: ignore[arg-type]
            t_min_crane=float(d.get("t_min_crane", defaults.t_min_crane)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a PPO training run.

    Parameters
    ----------
    steps : int
        Total training timesteps (default 100 000).
    n_envs : int
        Number of parallel environments (default 4).
    gamma : float
        Discount factor (default 0.99).
    save_path : str
        Where to save the trained model (default ``models/ppo_AntiPendulumEnv.zip``).
    seed : int or None
        Random seed passed to PPO for reproducibility (default None - non-deterministic).
    ent_coef : float
        Entropy bonus coefficient (default 0.0). Increase to 0.005-0.01 to encourage
        exploration and reduce sensitivity to random seed.
    learning_rate : float
        Adam learning rate (default 3e-4, the SB3 default).
    clip_range : float
        PPO clipping parameter (default 0.2). Lower values give more conservative updates.
    n_steps : int
        Timesteps collected per environment before each gradient update (default 2048,
        the SB3 default). Increasing to 8192 gives ~11 episodes per update instead of
        ~3, producing more stable gradient estimates for long-horizon tasks.
    randomize_start : bool
        If True, sample the initial pendulum speed uniformly from
        ``[min_speed, abs(start_speed)]`` with random sign each episode.
        Discourages the policy from overfitting to a single starting trajectory
        (default False - deterministic, matching evaluation behaviour).
    rail_limit : float
        Half-span of the crane rail in metres (default 10.0). The crane spans
        ``+-rail_limit``; an episode is truncated when ``|x| > rail_limit``.
        Reducing to e.g. 2.0 triggers earlier termination and tightens the
        credit-assignment gap for the terminal penalty.
    start_speed : float
        Initial pendulum speed passed to the environment (default 1.0). With
        ``randomize_start=True`` the actual per-episode speed is sampled from
        ``+-[min_speed, start_speed]``, so this acts as the upper bound of the
        training speed range.
    continuous_actions : bool
        If True, use a ``Box([-1], [1])`` action space so PPO can output any
        acceleration in ``[-acc, +acc]``.  If False, use ``Discrete(3)`` for
        Q-learning compatibility (default True).
    reward_limit : float
        Per-step reward threshold at which an episode is terminated as solved
        (default 50.0, effectively disabled). 0.0 is the theoretical maximum;
        setting to a large positive value (e.g. 50.0) disables early
        termination so the episode always runs to ``max_episode_steps``.
    max_episode_steps : int
        Maximum steps per episode enforced via a TimeLimit wrapper (default
        100). Replaces the previous hardcoded value of 3000; shorter episodes
        let the discount factor propagate rail-penalty credit meaningfully
        (``0.99^100 ≈ 0.37`` vs ``0.99^3000 ≈ 10^-13``).
    """

    steps: int = 100_000
    n_envs: int = 4
    gamma: float = 0.99
    save_path: str = "models/ppo_AntiPendulumEnv.zip"
    seed: int | None = None
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    n_steps: int = 2048
    randomize_start: bool = False
    rail_limit: float = 10.0
    start_speed: float = 1.0
    continuous_actions: bool = True
    reward_limit: float = 50.0
    max_episode_steps: int = 1000

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> TrainingConfig:
        """Instantiate from a mapping, filling missing keys with defaults.

        Parameters
        ----------
        d : dict[str, object]
            Mapping of field names to values. Unknown keys are ignored.

        Returns:
        -------
        TrainingConfig
            Populated instance.
        """
        defaults = cls()
        seed_raw = d.get("seed", defaults.seed)
        return cls(
            steps=int(d.get("steps", defaults.steps)),  # type: ignore[arg-type,call-overload]
            n_envs=int(d.get("n_envs", defaults.n_envs)),  # type: ignore[arg-type,call-overload]
            gamma=float(d.get("gamma", defaults.gamma)),  # type: ignore[arg-type]
            save_path=str(d.get("save_path", defaults.save_path)),
            seed=int(seed_raw) if isinstance(seed_raw, int) else None,
            ent_coef=float(d.get("ent_coef", defaults.ent_coef)),  # type: ignore[arg-type]
            learning_rate=float(d.get("learning_rate", defaults.learning_rate)),  # type: ignore[arg-type]
            clip_range=float(d.get("clip_range", defaults.clip_range)),  # type: ignore[arg-type]
            n_steps=int(d.get("n_steps", defaults.n_steps)),  # type: ignore[arg-type,call-overload]
            randomize_start=bool(d.get("randomize_start", defaults.randomize_start)),
            rail_limit=float(d.get("rail_limit", defaults.rail_limit)),  # type: ignore[arg-type]
            start_speed=float(d.get("start_speed", defaults.start_speed)),  # type: ignore[arg-type]
            continuous_actions=bool(d.get("continuous_actions", defaults.continuous_actions)),
            reward_limit=float(d.get("reward_limit", defaults.reward_limit)),  # type: ignore[arg-type]
            max_episode_steps=int(d.get("max_episode_steps", defaults.max_episode_steps)),  # type: ignore[arg-type,call-overload]
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Parameters
    ----------
    reward : RewardConfig
        Reward weights (defaults to canonical PPO values).
    training : TrainingConfig
        Training hyperparameters (defaults to script defaults).
    config_source : str or None
        Path of the YAML file this config was loaded from (default None).
    """

    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    config_source: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, object], *, config_source: str | None = None) -> ExperimentConfig:
        """Instantiate from a nested mapping.

        Parameters
        ----------
        d : dict[str, object]
            Mapping with optional ``reward`` and ``training`` sub-dicts.
            A ``config_source`` key in ``d`` is also restored when present.
        config_source : str or None
            Path to record as the origin; takes precedence over the key in ``d``
            (default None).

        Returns:
        -------
        ExperimentConfig
            Populated instance.
        """
        reward_raw = d.get("reward", {})
        training_raw = d.get("training", {})
        resolved_source: str | None = config_source
        if resolved_source is None:
            raw = d.get("config_source")
            if isinstance(raw, str):
                resolved_source = raw
        return cls(
            reward=RewardConfig.from_dict(reward_raw if isinstance(reward_raw, dict) else {}),
            training=TrainingConfig.from_dict(training_raw if isinstance(training_raw, dict) else {}),
            config_source=resolved_source,
        )


def load_experiment_config(config_path: str | Path | None) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to the YAML config file. Returns an all-default
        :class:`ExperimentConfig` when ``None``.

    Returns:
    -------
    ExperimentConfig
        Loaded configuration; missing YAML keys fall back to dataclass defaults.

    Raises:
    ------
    FileNotFoundError
        When ``config_path`` is not ``None`` but the file does not exist.
    """
    if config_path is None:
        return ExperimentConfig()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw: dict[str, object] = yaml.safe_load(fh) or {}
    logger.info("Loaded experiment config from %s", path)
    return ExperimentConfig.from_dict(raw, config_source=str(path))


def _meta_path(model_path: str | Path) -> Path:
    """Return the sidecar path for a model file.

    Parameters
    ----------
    model_path : str or Path
        Path to the model ``.zip`` file.

    Returns:
    -------
    Path
        Sibling file with the same stem and ``_meta.json`` suffix.
    """
    p = Path(model_path)
    return p.parent / f"{p.stem}_meta.json"


def save_training_sidecar(model_path: str | Path, config: ExperimentConfig) -> Path:
    """Write a JSON sidecar alongside a saved model.

    Parameters
    ----------
    model_path : str or Path
        Path to the model ``.zip`` file.
    config : ExperimentConfig
        Experiment configuration to serialise.

    Returns:
    -------
    Path
        Path of the written sidecar file.
    """
    sidecar = _meta_path(model_path)
    payload = dataclasses.asdict(config)
    _ = sidecar.write_text(json.dumps(payload, indent=2))
    logger.info("Training sidecar written to %s", sidecar)
    return sidecar


def load_training_sidecar(model_path: str | Path) -> ExperimentConfig:
    """Read the JSON sidecar for a saved model.

    Parameters
    ----------
    model_path : str or Path
        Path to the model ``.zip`` file.

    Returns:
    -------
    ExperimentConfig
        Configuration stored in the sidecar.

    Raises:
    ------
    FileNotFoundError
        When the sidecar file does not exist alongside the model.
    """
    sidecar = _meta_path(model_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"No sidecar found for model {str(model_path)!r}. Re-train with the current scripts to generate one."
        )
    raw: dict[str, object] = json.loads(sidecar.read_text())
    return ExperimentConfig.from_dict(raw)
