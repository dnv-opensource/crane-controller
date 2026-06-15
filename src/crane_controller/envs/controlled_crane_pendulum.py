"""Gymnasium environment for the crane anti-pendulum task."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from py_crane.animation import AnimatePlayBackLines

from crane_controller.experiment_config import RewardConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.lines import Line2D
    from py_crane.boom import Wire
    from py_crane.crane import Crane

logger = logging.getLogger(__name__)

MIN_PLAYBACK_FRAMES = 2
POLAR_Z_TOLERANCE = 0.1


def _level(idx: int, val: float, categories: tuple[float, ...]) -> tuple[int, int]:
    """Determine the bucket index for a value given ordered categories.

    Parameters
    ----------
    idx : int
        Fallback index returned when `val` falls below the first category.
    val : float
        Value to classify.
    categories : tuple[float, ...]
        Ordered category boundaries.

    Returns
    -------
    tuple[int, int]
        ``(bucket_index, error_flag)`` where *error_flag* is `idx` when the
        value is below the first category and ``0`` otherwise.
    """
    if val < categories[0]:
        return 0, idx
    return next(
        ((i, 0) for i, category in enumerate(categories) if val <= category),
        (len(categories) - 1, idx),
    )


# Observation is either a discrete tuple or a continuous ndarray
AntiPendulumObs = tuple[int, ...] | np.ndarray


class AntiPendulumEnv(gym.Env[AntiPendulumObs, int | np.ndarray]):
    """Environment for a py-crane-based anti-pendulum task.

    Uses the matplotlib-based animation module from py-crane.

    Parameters
    ----------
    crane : Callable[..., Crane]
        Factory callable that creates the crane object.
    acc : float, optional
        Acceleration magnitude applied to the crane (default 0.1).
    start_speed : float, optional
        Initial pendulum speed for each episode (default 1.0).
    randomize_start : bool, optional
        If True, sample the initial pendulum speed uniformly from
        ``[min_speed, abs(start_speed)]`` with random sign each reset.
        Discourages the policy from overfitting to a single starting trajectory
        (default False - deterministic).
    render_mode : str, optional
        One of the modes listed in ``metadata["render_modes"]``
        (default ``"none"``).
    rail_limit : float, optional
        Half-span of the crane rail in metres (default 10.0). The crane spans
        ``+-rail_limit``; an episode is truncated when ``|x| > rail_limit``.
    seed : int or None, optional
        Seed for repeatable random numbers (default None).
    reward_limit : float, optional
        Reward at which an episode is terminated and the anti-pendulum is
        deemed successful (default 50.0).
    dt : float, optional
        Simulation time step (default 1.0).
    discrete : dict[str, tuple[float | int, ...]] or None, optional
        When provided, activates discrete observation mode with the given
        category boundaries. Expected keys: ``"angles"``, ``"pos"``,
        ``"speed"``, ``"distance"``, ``"sector"`` (default None).
    reward_fac : RewardConfig, optional
        Weights for the reward contributions. Defaults to the canonical PPO
        values - see :class:`.RewardConfig`.
    continuous_actions : bool, optional
        If True, the action space is ``Box([-1], [1])`` and an action value
        in ``[-1, 1]`` is scaled by ``acc`` to produce the crane acceleration.
        If False, the action space is ``Discrete(3)`` with mapping
        ``0=-acc, 1=0, 2=+acc`` (Q-agent compatible, default False).
    """

    metadata: ClassVar[dict[str, object]] = {  # pyright: ignore[reportIncompatibleVariableOverride]  # Gymnasium metadata typing is loose
        "render_modes": (
            "none",
            "play-back",
            "data",
            "reward-tracking",
            "plot",
            "agent",
        ),
        "interval": 100,
        "show-len-1": False,
        "x-max": None,
    }

    def __init__(  # noqa: PLR0913 - environment API needs explicit parameters
        self,
        crane: Callable[..., Crane],
        acc: float = 0.1,
        start_speed: float = 1.0,
        randomize_start: bool = False,  # noqa: FBT001, FBT002
        render_mode: str = "none",
        rail_limit: float = 10.0,
        seed: int | None = None,
        reward_limit: float = 50.0,
        dt: float = 1.0,
        discrete: dict[str, tuple[float | int, ...]] | None = None,
        reward_fac: RewardConfig | None = None,
        continuous_actions: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the anti-pendulum environment.

        See the class docstring for parameter descriptions.
        """
        self.crane_maker = crane
        self.crane: Crane = crane()
        wire = self.crane.boom_by_name("wire")
        assert wire is not None, "Need a crane wire!"
        self.wire: Wire = wire  # type: ignore[assignment]  # boom_by_name returns Boom; at runtime this is Wire
        assert render_mode in self.metadata["render_modes"], f"render_mode: {render_mode}"  # type: ignore[operator]  # metadata values are typed as object
        self.render_mode = render_mode
        self.reward_fac: RewardConfig = reward_fac if reward_fac is not None else RewardConfig()
        self.continuous_actions = continuous_actions
        self.reward_stats: list[list[float]] = []
        self._playback: list[list[float]] = []
        self.rewards: list[float] = []
        if render_mode == "reward-tracking":
            self._reward_point = self._reward_plot_init()
        elif render_mode == "plot":
            self.traces: dict[str, list[float]] = {"c_x": [], "c_v": [], "l_x": [], "l_v": [], "acc": [], "t_min": []}

        self.obeservation_space: spaces.Box | spaces.Discrete  # pyright: ignore[reportMissingTypeArgument]  # Discrete type arg not needed here
        # Continuous observations: crane position, crane velocity, wire polar angle, pure angular velocity theta_dot.
        self.min_speed = 0.1  # np.sqrt(2*reward_limit) # starting with less does not make sense (goal already reached)
        max_speed = np.sqrt(9.81 * self.wire.length)  # speed for pendulum at +/- 90 deg. Polar as deflection from -z
        if discrete is not None:
            self.observation_space, self.discrete = self._init_discrete(discrete)
        else:
            self.discrete = {}
            self.spaces_min = np.array((-rail_limit, -max_speed, 0.0, -max_speed), float)
            self.spaces_max = np.array((rail_limit, max_speed, np.pi, max_speed), float)
            self.observation_space = spaces.Box(self.spaces_min, self.spaces_max, shape=(4,), dtype=np.float64)

        self.nresets: int = 0
        self.acc = acc
        self.start_speed = start_speed
        self.randomize_start = randomize_start
        self.initial_speed: float = start_speed
        self.rail_limit = rail_limit
        self.figsize: tuple[float, float] = (-rail_limit, rail_limit)  # The matplotlib animation window
        self.reward_limit = reward_limit
        self.nsuccess: int = 0
        self.reward = 0.0  # a basic reward (pendulum energy + distance measure)
        self.dt = dt
        self._prev_theta_dot: float | None = None

        if continuous_actions:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # Discrete actions: 0 = -acc (left), 1 = 0 (coast), 2 = +acc (right)
            self.action_space = spaces.Discrete(3, start=0, seed=42, dtype=np.int64)
        self.action_to_acc = {0: -self.acc, 1: 0.0, 2: self.acc}
        self.steps: int = 0
        self.time: float = 0.0
        _ = super().reset(seed=seed)

    def _init_discrete(
        self,
        spec: dict[str, tuple[float | int, ...]],
    ) -> tuple[spaces.MultiDiscrete, dict[str, tuple[float | int, ...]]]:
        """Translate the observation-space spec into a MultiDiscrete space.

        Expected keys in `spec`::

            'angles'   - amplitude categories (converted to energy levels)
            'pos'      - load position (+/- x)
            'speed'    - load speed (+/- x)
            'distance' - distance categories from origin
            'sector'   - crane position sector (+/- x)

        Parameters
        ----------
        spec : dict[str, tuple[float | int, ...]]
            Mapping of observation dimension names to category boundaries.

        Returns
        -------
        tuple[spaces.MultiDiscrete, dict[str, tuple[float | int, ...]]]
            The constructed ``MultiDiscrete`` space and the updated spec
            (with ``'angles'`` replaced by ``'energies'``).
        """
        # We replace the angles with pendulum energy levels, which are easier to use for observation calculation
        observation_space = spaces.MultiDiscrete(np.array([len(spec[k]) for k in spec]))
        angles = spec.pop("angles")
        energies = [9.81 * self.wire.length * (1.0 - np.cos(np.radians(a))) for a in angles]
        spec["energies"] = tuple(energies)
        return (observation_space, spec)

    def _reward_plot_init(self, marker: str = "") -> Line2D:
        point = plt.plot(0, 0, marker)[0] if marker else plt.plot(0, 0)[0]
        plt.ion()
        plt.show()
        return point

    def _append_playback(self, time: float) -> None:
        """Append the current crane state to the playback buffer.

        Parameters
        ----------
        time : float
            Current simulation time.
        """
        if not len(self._playback):  # no records there yet
            self._playback.append([time])  # slot 0 for time
            for b in self.crane.booms():
                self._playback.append([b.end])  # type: ignore[list-item]  # TVector is compatible at runtime
        else:
            self._playback[0].append(time)
            for i, b in enumerate(self.crane.booms()):
                self._playback[i + 1].append(b.end)  # type: ignore[arg-type]  # TVector is compatible at runtime

    def show_animation(self) -> None:
        """Show the playback animation of the current episode recording."""
        if len(self._playback[0]) < MIN_PLAYBACK_FRAMES and not AntiPendulumEnv.metadata["show-len-1"]:
            return
        data = [np.array(col, float) for col in self._playback]
        ani = AnimatePlayBackLines(
            data=data,
            lw=(5, 1),
            figsize=(10, 10),
            interval=int(AntiPendulumEnv.metadata["interval"]),  # type: ignore[call-overload]  # metadata value is int at runtime
            title="Anti-Pendulum episode",
        )
        ani.do_animation()

    def show_plot(self, episode: int, save_path: str | None = None) -> None:
        """Plot detailed traces for a single episode.

        Parameters
        ----------
        episode : int
            Episode number used in the plot title.
        save_path : str or None, optional
            If set, save the figure to this path and close it instead of calling
            ``plt.show()`` (default None — interactive display).
        """
        if not self.traces["l_v"]:
            return
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(16, 21), sharex=True)
        times = self.dt * np.arange(len(self.traces["c_x"]))
        damping = self.traces["l_v"][0] * np.exp(-times / self.wire.damping_time)
        ax1.plot(times, self.traces["l_x"], label="load angle", color="blue")
        ax2.plot(times, self.traces["l_v"], label="load speed", color="red")
        ax2.plot(times, damping, label="damping", color="green")
        ax3.plot(times, self.traces["c_x"], label="crane pos", color="blue")
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="origin")
        ax4.plot(times, self.traces["c_v"], label="crane speed", color="red")
        ax5.plot(times[: len(self.rewards)], self.rewards, label="rewards")
        ax6.plot(times, self.traces["acc"], label="x-acceleration", color="green")
        ax7.plot(times[: len(self.traces["t_min"])], self.traces["t_min"], label="t_min", color="purple")
        ax7.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax7.set_xlabel("time [s]")
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7):
            _ = ax.legend()
        _ = plt.suptitle(
            f"Detailed plot of episode {episode}, reward:{self.reward}, start_speed:{self.initial_speed:.3f}"  # pyright: ignore[reportUnknownMemberType]
        )
        fig.tight_layout()
        if save_path is not None:
            from pathlib import Path  # noqa: PLC0415
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        for key in self.traces:
            self.traces[key] = []
        self.rewards = []

    def _get_continuous_obs(self) -> tuple[np.ndarray, int]:
        """Return continuous observations and an out-of-bounds error flag.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(observation, error_flag)`` where *error_flag* is ``0`` when all
            values are within bounds, or the 1-based index of the first
            out-of-bounds dimension.
        """
        err = 0
        obs = np.array(
            (
                self.crane.position[0],
                self.crane.velocity[0],
                self.wire.boom[1],
                (self.wire.cm_v[0] - self.wire.origin_v[0]) / self.wire.length,  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            ),
            float,
        )

        for i, (_obs, _min, _max) in enumerate(zip(obs, self.spaces_min, self.spaces_max, strict=True)):
            if not _min <= _obs <= _max:
                err = i + 1

        return obs, err

    def _get_discrete_obs(self, energy: float) -> tuple[int, ...]:
        """Return the discrete observation tuple from the current crane state.

        Parameters
        ----------
        energy : float
            Current pendulum energy.

        Returns
        -------
        tuple[int, ...]
            Discretised observation ``(energy_level, side, speed_sign,
            distance_level, sector)``.
        """
        energy_level, _ = _level(1, energy, self.discrete["energies"])
        distance_level, _ = _level(3, abs(self.crane.position[0]), self.discrete["distance"])
        return (
            energy_level,
            int(self.wire.end[0] - self.wire.origin[0] < 0.0),
            int(self.wire.cm_v[0] < 0.0),  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            distance_level,
            int(self.crane.position[0] < 0.0),
        )

    def _get_obs(self, acc: float = 0.0) -> tuple[np.ndarray | tuple[int, ...], float, int]:
        """Compute the current observation and reward from the crane state.

        In discrete mode the observation keys are::

            'energies' - energy categories of the load
            'side'     - current side of the load (+/- x)
            'distance' - distance categories from origin
            'sector'   - crane position sector (+/- x)

        Returns
        -------
        tuple[np.ndarray | tuple[int, ...], float, int]
            ``(observation, reward, error_flag)``.
        """
        energy = 9.81 * self.wire.end[2] + 0.5 * np.dot(self.wire.cm_v, self.wire.cm_v)  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        if self.start_speed != 0.0:  # anti-pendulum mode
            energy = -energy
        if np.sign(self.crane.position[0]) == np.sign(self.crane.velocity[0]):  # moving away from origo
            positional = -self.wire.length * (abs(self.crane.position[0]) + self.crane.velocity[0] ** 2 / self.acc)
        else:
            positional = 0.0  # if the crane moves towards the origo we do not subtract reward
        position = -abs(self.crane.position[0])
        acc_penalty = -abs(acc)
        rc = self.reward_fac
        self.reward = (
            rc.energy * energy
            + rc.positional * positional
            + rc.time * (-self.time)
            + rc.position * position
            + rc.acceleration * acc_penalty
        )
        theta = self.wire.boom[1]
        theta_dot = (self.wire.cm_v[0] - self.wire.origin_v[0]) / self.wire.length  # pyright: ignore[reportUnknownMemberType]
        theta_ddot = (theta_dot - self._prev_theta_dot) / self.dt if self._prev_theta_dot is not None else 0.0
        self._prev_theta_dot = theta_dot
        self.reward += (  # pyright: ignore[reportUnknownMemberType]
            rc.angle * (-(theta**2))
            + rc.angular_velocity * (-(theta_dot**2))
            + rc.crane_velocity * (-(self.crane.velocity[0] ** 2))
            + rc.crane_acceleration * (-(acc**2))
            + rc.angular_acceleration * (-(theta_ddot**2))
            + rc.t_min_crane * (-self._t_min_crane())
        )

        obs: tuple[int, ...] | np.ndarray
        if len(self.discrete):
            obs = self._get_discrete_obs(energy)
            err = 0

        else:
            obs, err = self._get_continuous_obs()

        if self.render_mode == "plot":
            self.traces["c_x"].append(self.crane.position[0])
            self.traces["c_v"].append(self.crane.velocity[0])
            self.traces["l_x"].append(self.wire.c_m[0])
            self.traces["l_v"].append(self.wire.cm_v[0])  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            self.traces["acc"].append(acc)
            self.traces["t_min"].append(self._t_min_crane())

        return (obs, self.reward, err)  # pyright: ignore[reportUnknownMemberType]

    def _t_min_crane(self) -> float:
        """Minimum time for the crane to reach x=0 at rest under bang-bang control.

        Returns
        -------
        float
            Optimal time-to-origin in seconds; zero when crane is already at rest
            at the origin.
        """
        x0 = self.crane.position[0]
        v0 = self.crane.velocity[0]
        a = self.acc
        if (x0 >= 0 and v0 >= 0) or (x0 <= 0 and v0 <= 0):  # moving away from origin
            return (abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a + 0.5 * v0**2))) / a
        # moving toward origin
        if abs(x0) >= 0.5 * v0**2 / a:  # no overshoot
            return (-abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a + 0.5 * v0**2))) / a
        # overshoot
        return (abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a - 0.5 * v0**2))) / a

    def low_reward(self) -> float:
        """Return the lowest possible reward for the current mode.

        Returns
        -------
        float
            Zero in start-pendulum mode, or the negative maximum pendulum
            energy in stop-pendulum mode.
        """
        if self.start_speed == 0.0:
            return 0.0
        return -float(self.discrete["energies"][-1])

    def _get_info(self, reward: float, steps: int) -> dict[str, float | int]:
        return {
            "steps": steps,
            "reward": reward,
            "t_min": self._t_min_crane(),
            "x_pos": self.crane.position[0],
            "x_vel": self.crane.velocity[0],
            "energy": 0.5 * float(self.wire.cm_v[0]) ** 2,  # pyright: ignore[reportUnknownMemberType]
            "theta": float(self.wire.boom[1]),
            "theta_dot": (float(self.wire.cm_v[0]) - float(self.wire.origin_v[0])) / float(self.wire.length),  # pyright: ignore[reportUnknownMemberType]
        }

    def reset_crane(self) -> None:
        """Reset the crane to its initial physical state.

        Sets position, velocity, torque, and force to zero, recalculates
        statics and dynamics, and relaxes the pendulum wire.
        """
        self.crane.position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.d_velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.torque = np.array((0, 0, 0), dtype=np.float64)
        self.crane.force = np.array((0, 0, 0), dtype=np.float64)
        self.crane.current_time = 0.0  # used to make current_time from do_step known for the whole crane
        self.crane.boom0.update_child()
        self.crane.calc_statics_dynamics(None)
        self.wire.pendulum_relax()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[tuple[int, ...] | np.ndarray, dict[str, float | int]]:
        """Reset the environment for a new episode.

        Args:
            seed (int): Optional random seed (default None).
            options (dict[str, object]): Optional additional arguments to super().reset(). Default None.

        Returns
        -------
            tuple[tuple[int, ...] | np.ndarray, dict[str, float | int]]
            Initial observation and info dict.
        """
        self.reset_crane()

        if self.nresets <= 0:  # reset during instantiation. Initialize
            if self.render_mode == "data":
                self._reward_point = self._reward_plot_init("b.")

        else:  # reset between episodes. Data are available
            self.reward_stats.append([self.steps, self.reward])  # pyright: ignore[reportUnknownMemberType]
            if self.render_mode == "data":
                self._reward_point.set_data([r[0] for r in self.reward_stats], [r[1] for r in self.reward_stats])
                plt.pause(1e-10)
            elif self.render_mode == "play-back" and len(self._playback):
                self.show_animation()
                self._playback = []
            elif self.render_mode == "plot":
                self.show_plot(self.nresets)

        _ = super().reset(seed=seed, options=options)

        self.nresets += 1
        if self.start_speed == 0.0:  # run in 'start' mode, learning how to start the pendulum action
            assert self.wire.cm_v[0] == 0.0, f"Load speed expected zero. Found {self.wire.cm_v[0]}"  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        elif self.randomize_start:
            speed = self.np_random.uniform(self.min_speed, abs(self.start_speed))
            sign = 1.0 if self.np_random.random() > 0.5 else -1.0  # noqa: PLR2004
            self.wire.cm_v[0] = speed * sign  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        else:
            self.wire.cm_v[0] = self.start_speed  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        self.initial_speed = float(self.wire.cm_v[0])  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        self._prev_theta_dot = None
        obs, self.reward, _ = self._get_obs()
        if self.render_mode == "play-back":
            self._append_playback(0.0)
        self.steps = 0
        self.time = 0.0
        info = self._get_info(self.reward, self.steps)
        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action : int or np.ndarray
            Discrete mode: index 0 = -acc, 1 = 0, 2 = +acc.
            Continuous mode: float in ``[-1, 1]`` scaled by ``acc``.

        Returns
        -------
        tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]
            ``(observation, reward, terminated, truncated, info)``.
        """
        if self.continuous_actions:
            acc = float(np.asarray(action).flat[0]) * self.acc
        else:
            action_idx = int(action)
            if action_idx not in self.action_to_acc:
                action_idx += 1
            acc = self.action_to_acc[action_idx]
        self.crane.d_velocity[0] = acc
        self.steps += 1
        _ = self.crane.do_step(self.time, self.dt)
        self.time += self.dt

        obs, self.reward, truncated = self._get_obs(acc)
        if truncated and self.reward_fac.terminal_penalty != 0.0:
            self.reward += self.reward_fac.terminal_penalty
        if self.render_mode != "none":
            self.rewards.append(self.reward)

        if self.render_mode == "play-back":
            self._append_playback(self.steps)
        elif self.render_mode == "reward-tracking":
            _ = self._reward_point.set_data(list(range(len(self.rewards))), self.rewards)
            _ = plt.xlim((0, len(self.rewards)))
            _ = plt.ylim((min(self.rewards), max(self.rewards)))
            plt.pause(1e-10)
        terminated = self.reward > self.reward_limit
        if terminated:
            self.nsuccess += 1
        info = self._get_info(self.reward, self.steps)
        if truncated > 0:
            info["crash"] = True
        return obs, self.reward, terminated, (truncated > 0), info

    def render(self, save_path: str | None = None) -> None:
        """Render the current episode.

        Parameters
        ----------
        save_path : str or None, optional
            If set and render_mode is ``"plot"``, save the figure to this path
            instead of showing it interactively (default None).
        """
        if self.render_mode == "play-back":
            self.show_animation()
        elif self.render_mode == "plot":
            self.show_plot(self.nresets, save_path=save_path)
