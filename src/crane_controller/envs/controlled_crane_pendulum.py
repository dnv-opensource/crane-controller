"""Gymnasium environment for the crane anti-pendulum task."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from component_model.utils.transform import cartesian_to_spherical
from gymnasium import spaces
from py_crane.animation import AnimatePlayBackLines
from py_crane.boom import Wire

from crane_controller.crane_factory import build_crane
from crane_controller.experiment_config import RewardConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.lines import Line2D
    from py_crane.crane import Crane

LOGGER = logging.getLogger(__name__)

MIN_PLAYBACK_FRAMES = 2
POLAR_Z_TOLERANCE = 0.1
EPS = 1e-10


def make_discretization(discrete: str) -> dict[str, tuple[float, ...]]:
    """Prepare and return a discretization from a name.

    Args:
        discrete: the name of the discretization.

    Returns:
        discretization dict
    """
    if discrete == "energy":  # oriented along energy and distance with binary 'regions'
        return {
            "angle": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "distance": (0.0, 0.5, 1.0, 2.0),
            "pos": (0.0, 1.0),
            "speed": (0.0, 1.0),
            "c-pos": (0.0, 1.0),
            "c-speed": (0.0, 1.0),
            "avg-acc": tuple(np.linspace(-1.25, 1.25, 11)),
        }
    if discrete == "phase-min":  # similar to 'phase' but much less categories
        return {
            "angle": tuple(np.radians((-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0))),
            "speed": tuple(np.linspace(-4.0, 4.0, 5)),  # only x-component to preserve sign!
            "c-pos": (-2.0, -0.5, -0.125, 0, 0.125, 0.5, 2.0),
            "c-speed": (-2.0, -0.5, -0.125, 0, 0.125, 0.5, 2.0),
            "avg-acc": (-1.1, -1, -0.5, 0.0, 0.5, 1.0, 1.1),
        }
    # oriented along 'phase' of load and crane. Used also as fallback discretization
    return {
        "angle": tuple(np.radians((-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0))),
        "speed": tuple(np.linspace(-5.0, 5.0, 11)),  # only x-component to preserve sign!
        "c-pos": (-2.0, -1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0, 2.0),
        "c-speed": (-2.0, -1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0, 2.0),
        "avg-acc": tuple(np.linspace(-1.25, 1.25, 11)),
    }


@dataclass(kw_only=True, frozen=True, slots=True)
class AntiPendulumConfig:
    """Configuration parameters for AntiPendulum environment.

    Args:
        length: the length of the crane wire (and the pedestal)
        mass: mass of the crane load
        q_factor: the damping factor of the pendulum action
        acc: Acceleration magnitude applied to the crane.
        start_speed: Fixed start speed in m/s. A negative value causes a random speed
           in the range ``[-|start_speed|, |start_speed|]`` each episode
        randomize_start: Optional randomize the start speed within +/- start_speed
        render_mode: One of the modes listed in ``metadata["render_modes"]``
        rail_limit: Half-span of the crane rail in metres (default 10.0). The crane spans
            ``+-rail_limit``; within PPO an episode is truncated when ``|x| > rail_limit``.
        seed: Seed for repeatable random numbers.
        reward_limit: Reward at which an episode is terminated and the anti-pendulum is deemed successful
        dt: Simulation time step
        discrete: When provided, activates discrete observation mode with the given named category set.
        reward_fac: Weights between reward contributions
        discount: discount factor for acceleration history to include in (discrete) observation
        continuous_actions: If True, the action space is ``Box([-1], [1])`` and an action value
            in ``[-1, 1]`` is scaled by ``acc`` to produce the crane acceleration.
            If False, the action space is ``Discrete(3)`` with mapping``0=-acc, 1=0, 2=+acc`` (Q-agent compatible).
        discretization: dict of discretization tuples, relevant for discrete != 'none'
        coasting: Allow coasting as one of the actions (acceleration = 0.0)
    """

    # crane configuration
    length: float = 10.0
    mass: float = 1.0
    q_factor: float = 50.0
    # environment for use by agent
    acc: float = 0.1
    start_speed: float = 1.0
    randomize_start: bool = False
    render_mode: str = "none"
    rail_limit: float = 10.0
    seed: int | None = None
    reward_limit: float | None = None
    dt: float = 1.0
    discrete: str = "none"
    discretization: dict[str, tuple[float,...]] | None = None
    reward_fac: RewardConfig | None = None
    continuous_actions: bool = False
    discount: float = 0.8
    coasting: bool = True


class AntiPendulumEnv(gym.Env[tuple[int, ...] | np.ndarray, int]):
    """Environment for a py-crane-based anti-pendulum task.

    Uses the matplotlib-based animation module from py-crane.
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

    def __init__(self, crane: Callable[..., Crane] | None = None, conf: AntiPendulumConfig | None = None) -> None:
        """Initialize the anti-pendulum environment.

        Args:
            crane: Factory callable that creates the crane object.
            conf: Configuration parameters as dataclass. See AntiPendulumConfig.
        """
        if crane is None:
            self.crane_maker = build_crane
        else:
            self.crane_maker = crane
        self.conf = AntiPendulumConfig() if conf is None else conf
        self.render_mode: str | None = self.conf.render_mode  # gymnasium convention: expose as direct attribute
        self.crane: Crane = self.crane_maker(length=self.conf.length, mass=self.conf.mass, q_factor=self.conf.q_factor)
        self.wire: Wire = self.crane.boom_by_name("wire")  # type: ignore[assignment]  # Wire is a sub-class of Boom
        assert isinstance(self.wire, Wire), "Need a crane wire!"
        assert self.conf.render_mode in AntiPendulumEnv.metadata["render_modes"], (  # type: ignore[operator]  # metadata values are typed as object
            f"render_mode: {self.conf.render_mode}"
        )
        self.reward_fac = (
            self.conf.reward_fac
            if isinstance(self.conf.reward_fac, RewardConfig)
            else (RewardConfig() if self.conf.reward_fac is None else RewardConfig.from_dict(self.conf.reward_fac))
        )
        self.reward_stats: dict[str, list[float]] = {
            "steps": [],  # number of steps in episode
            "reward": [],  # final reward
            "status": [],  # -1:truncated, 0: max steps, 1: terminated
            "start": [],  # start reward
            "relaxation": [],  # reward relaxation time (exponential model)
        }
        self._playback: list[list[float]] = []
        self.rewards: list[float] = []
        if self.conf.render_mode == "reward-tracking":
            self._reward_point = self._reward_plot_init()
        elif self.conf.render_mode == "plot":
            self.traces: dict[str, list[float]] = {"c_x": [], "c_v": [], "l_x": [], "l_v": [], "acc": []}

        self.observation_space: spaces.Box | spaces.Discrete  # pyright: ignore[reportMissingTypeArgument]  # Discrete type arg not needed here
        self.discrete: dict[str, tuple[float, ...]]
        # Continuous observations are crane position, crane velocity, wire polar angle, and load x-velocity.
        max_speed = np.sqrt(9.81 * self.wire.length)  # speed for pendulum at +/- 90 deg. Polar as deflection from -z
        self.acc_hist: float = 0.0  # used for acceleration history discretization

        if self.conf.discrete != "none":
            self.observation_space, self.discrete = self.init_discrete()  # type: ignore[assignment]
        else:
            self.discrete = {}
            self.spaces_min = np.array([-self.conf.rail_limit, -max_speed, 0.0, -max_speed], dtype=np.float64)
            self.spaces_max = np.array([self.conf.rail_limit, max_speed, np.pi, max_speed], dtype=np.float64)
            self.observation_space = spaces.Box(self.spaces_min, self.spaces_max, shape=(4,), dtype=np.float64)  # type: ignore[reportIncompatibleVariableOverride]

        self.tau_max = self.distance_max / self.conf.acc / self.conf.dt  # time with min. speed from 0 to end

        self.nresets: int = -1
        self.seed = self.conf.seed  # might be changed if training is continued from saved values
        _ = super().reset(seed=self.seed)
        self.initial_speed: float = self.conf.start_speed
        self.figsize: tuple[float, float] = (-self.conf.rail_limit, self.conf.rail_limit)  # animation window
        self.nsuccess: int = 0
        self.reward = 0.0  # a basic reward (pendulum energy + distance measure)

        if self.conf.continuous_actions:
            self.action_space = spaces.Box(low=-self.conf.acc, high=self.conf.acc, shape=(1,), dtype=np.float32)  # type: ignore[assignment]
        elif self.conf.coasting:
            # Discrete actions: 0 = -acc (left), 1 = 0 (coast), 2 = +acc (right)
            self.action_space = spaces.Discrete(3, start=0, seed=self.conf.seed, dtype=np.int64)
            self.action_to_acc = {0: -self.conf.acc, 1: 0.0, 2: self.conf.acc}
        else:
            self.action_space = spaces.Discrete(2, start=0, seed=self.conf.seed, dtype=np.int64)
            self.action_to_acc = {0: -self.conf.acc, 1: self.conf.acc}
        self.steps: int = 0
        self.time: float = 0.0
        self.obs: tuple[int, ...] | np.ndarray  # previous observation
        self.energy0: float = 0.0  # save the initial energy (set by reset())

    def init_discrete(self) -> tuple[spaces.MultiDiscrete, dict[str, tuple[float, ...]]]:
        """Translate the observation-space spec into a MultiDiscrete space.

        See AntiPendulumConfig for pre-defined discretizations.

        Returns:
        -------
            The constructed ``MultiDiscrete`` space and the spec
        """
        self.acc_hist = 0.0
        if self.conf.discretization is not None:
            _spec = self.conf.discretization.copy()
        elif self.conf.discrete == "energy":
            base_spec = make_discretization(self.conf.discrete).copy()
            # We replace the angle with pendulum energy levels, which are easier to use for observation calculation
            angle = base_spec.pop("angle")
            energy = [9.81 * self.wire.length * (1.0 - np.cos(np.radians(a))) for a in angle]
            _spec = {"energy": tuple(energy)}
            _spec.update(base_spec)
        elif self.conf.discrete in ("phase", "phase-min"):
            _spec = make_discretization(self.conf.discrete).copy()
        else:
            raise KeyError(f"Unknown discretization spec {self.conf.discrete} for discretization") from None

        return (spaces.MultiDiscrete(np.array([len(_spec[k]) for k in _spec])), _spec)

    @property
    def energy_max(self) -> float:
        """Return the maximum energy as property."""
        if self.conf.discrete == "energy":
            return self.discrete["energy"][-1]
        if self.conf.discrete in ("phase", "phase-min"):
            return 0.5 * self.discrete["speed"][-1] ** 2
        return 0.5 * make_discretization("phase")["speed"][-1] ** 2

    @property
    def distance_max(self) -> float:
        """Return the max. distance as property."""
        if self.conf.discrete == "energy" and "distance" in self.discrete:
            return self.discrete["distance"][-1]
        if self.conf.discrete in ("phase", "phase-min") and "c-pos" in self.discrete:
            return self.discrete["c-pos"][-1]
        return make_discretization("phase")["c-pos"][-1]

    @property
    def speed_max(self) -> float:
        """Return the maximum speed as property."""
        if self.conf.discrete == "energy":
            return self.distance_max / self.conf.dt / 10
        if self.conf.discrete in ("phase", "phase-min"):
            return self.discrete["c-speed"][-1]
        return make_discretization("phase")["c-speed"][-1]

    def _reward_plot_init(self, marker: str = "") -> Line2D:
        point = plt.plot(0, 0, marker)[0] if marker else plt.plot(0, 0)[0]
        plt.ion()
        plt.show()
        return point

    def _append_playback(self, time: float) -> None:
        """Append the current crane state to the playback buffer.

        Args:
            time (float): Current simulation time.
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

    def show_plot(self, episode: int = -1, save_path: str | None = None) -> None:
        """Plot detailed traces for a single episode.

        Args:
            episode: Episode number used in the plot title.
            save_path: If set, save the figure to this path and close it instead of calling ``plt.show()``
        """
        if not self.traces["l_v"]:
            return
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
        times = self.conf.dt * np.arange(len(self.traces["c_x"]))
        damping = self.traces["l_v"][0] * np.exp(-times / (2 * self.wire.damping_time))
        ax1.plot(times, self.traces["l_x"], label="load angle", color="blue")
        ax2.plot(times, self.traces["l_v"], label="load speed", color="red")
        ax2.plot(times, damping, label="natural damping +", color="green", linestyle="--")
        ax2.plot(times, -damping, label="natural damping −", color="green", linestyle="--")  # noqa: RUF001
        ax3.plot(times, self.traces["c_x"], label="crane pos", color="blue")
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="origin")
        ax4.plot(times, self.traces["c_v"], label="crane speed", color="red")
        ax5.plot(times[: len(self.rewards)], self.rewards, label="rewards")
        ax6.plot(times, self.traces["acc"], label="x-acceleration", color="green")
        ax6.set_xlabel("time [s]")
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            _ = ax.legend(loc="upper right")
        _ = plt.suptitle(
            f"Detailed plot of episode {episode}, reward:{self.reward}, start_speed:{self.initial_speed:.3f}"
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

    def _get_continuous_obs(self) -> tuple[np.ndarray, int]:
        """Return continuous observations and an out-of-bounds error flag.

        Returns:
        -------
            tuple[np.ndarray, int]: ``(observation, error_flag)`` where *error_flag* is ``0`` when all
            values are within bounds, or the 1-based index of the first out-of-bounds dimension.
        """
        err = 0
        self.obs = np.array(
            (
                self.crane.position[0],
                self.crane.velocity[0],
                self.wire.boom[1],
                (self.wire.cm_v[0] - self.wire.origin_v[0]) / self.wire.length,  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            ),
            float,
        )

        for i, (_obs, _min, _max) in enumerate(zip(self.obs, self.spaces_min, self.spaces_max, strict=True)):
            if not _min <= _obs <= _max:
                err = i + 1

        return self.obs, err

    def _get_discrete_obs(self, energy: float, acc: float) -> tuple[tuple[int, ...], bool]:
        """Return the discrete observation tuple from the current crane state.

        Args:
            energy:  Current pendulum energy.
            acc: current acceleration command

        Returns:
            Discretised observation as tuple of integers according to discretization definition + truncation (bool).
        """
        self.acc_hist = self.conf.discount * self.acc_hist + (1.0 - self.conf.discount) * acc
        if self.conf.discrete == "energy":
            obs = [
                _level(energy, self.discrete["energy"]),
                _level(abs(self.crane.position[0]), self.discrete["distance"]),
                int(self.wire.end[0] - self.wire.origin[0] < 0.0),
                int(self.wire.cm_v[0] < 0.0),  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
                int(self.crane.position[0] < 0.0),
                int(self.crane.velocity[0] < 0.0),
                _level(self.acc_hist, self.discrete["avg-acc"]),
            ]
        elif self.conf.discrete in ("phase", "phase-min"):
            angle = np.pi - self.wire.boom[1]
            obs = [
                _level(angle if self.wire.direction[0] >= 0 else -angle, self.discrete["angle"]),
                _level(self.wire.cm_v[0], self.discrete["speed"]),  # only x-component, to keep sign!
                _level(self.crane.position[0], self.discrete["c-pos"]),
                _level(self.crane.velocity[0], self.discrete["c-speed"]),
                _level(self.acc_hist, self.discrete["avg-acc"]),
            ]
        else:
            raise ValueError(f"Unknown discretization {self.conf.discrete}.") from None
        trunc = any(i < 0 for i in obs)
        return (tuple(obs), trunc)

    def _get_obs(self, acc: float = 0.0) -> tuple[np.ndarray | tuple[int, ...], float, int]:
        """Compute the current observation, the reward and the truncation flag from the crane state.

        In discrete mode the observation keys are as defined in .DEFAULT_DISCRETE

        Args:
            acc (float): Acceleration used to get to this state (for use in traces)

        Returns:
        -------
            tuple[np.ndarray | tuple[int, ...], float, int]: ``(observation, reward, truncate_flag)``.
        """
        energy = 9.81 * self.wire.end[2] + 0.5 * np.dot(self.wire.cm_v, self.wire.cm_v)  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        if self.conf.start_speed != 0.0:  # anti-pendulum mode
            energy = -energy
        if np.sign(self.crane.position[0]) == np.sign(self.crane.velocity[0]):  # moving away from origo
            positional = -self.wire.length * (abs(self.crane.position[0]) + self.crane.velocity[0] ** 2 / self.conf.acc)
        else:
            positional = 0.0  # if the crane moves towards the origo we do not subtract reward
        position = -abs(self.crane.position[0])
        acc_penalty = -abs(acc)
        rc = self.reward_fac
        self.reward = rc.energy * energy
        for rc_fac, rc_base in {
            rc.positional: positional,
            rc.time: (-self.time),
            rc.position: position,
            rc.acceleration: acc_penalty,
            rc.crane_velocity: self.crane.velocity[0] ** 2,
            rc.t_min_crane: self._t_min_crane(),
        }.items():
            if rc_fac != 0.0:
                self.reward += rc_fac * rc_base

        if len(self.discrete):
            self.obs, truncate = self._get_discrete_obs(energy, acc)
        else:
            self.obs, _truncate = self._get_continuous_obs()
            truncate = bool(_truncate)
        if self.conf.render_mode == "plot":
            self.traces["c_x"].append(self.crane.position[0])
            self.traces["c_v"].append(self.crane.velocity[0])
            self.traces["l_x"].append(self.wire.c_m[0])
            self.traces["l_v"].append(self.wire.cm_v[0])  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            self.traces["acc"].append(acc)

        return (self.obs, self.reward, truncate)

    def _t_min_crane(self) -> float:
        """Minimum time for the crane to reach x=0 at rest under bang-bang control.

        Returns:
        -------
        float
            Optimal time-to-origin in seconds; zero when crane is already at rest
            at the origin.
        """
        x0 = self.crane.position[0]
        v0 = self.crane.velocity[0]
        a = self.conf.acc
        if (x0 >= 0 and v0 >= 0) or (x0 <= 0 and v0 <= 0):  # moving away from origin
            return (abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a + 0.5 * v0**2))) / a
        # moving toward origin
        if abs(x0) >= 0.5 * v0**2 / a:  # no overshoot
            return (-abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a + 0.5 * v0**2))) / a
        # overshoot
        return (abs(v0) + 2.0 * np.sqrt(max(0.0, abs(x0) * a - 0.5 * v0**2))) / a

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
        """Reset the environment for a new episode. Used mainly by agent.

        Args:
            seed: Optional random seed (default None).
            init: Additional reset activities when initializing (e.g. plotting)
            options (dict[str, object]): Optional additional arguments to super().reset(). Default None.

        Returns:
        -------
            tuple[tuple[int, ...] | np.ndarray, dict[str, float | int]]: Initial observation and info dict.
        """
        self.reset_crane()

        if isinstance(options, dict) and "init" in options:  # reset during instantiation. Initialize
            init = options.pop("init")
            if init and self.conf.render_mode == "data":
                self._reward_point = self._reward_plot_init("b.")
                self.nresets = -1

        else:  # reset between episodes. Data are available
            self.reward_stats_calc(len(self.rewards))
            if self.conf.render_mode == "data":  # plot reward per step
                x = self.reward_stats["steps"]
                y = self.reward_stats["reward"]
                self._reward_point.set_data(x, y)
                _ = plt.xlim((min(x), max(x)))
                _ = plt.ylim((min(y), max(y)))
                plt.pause(1e-10)
            elif self.conf.render_mode == "play-back" and len(self._playback):  # crane animation
                self.show_animation()
                self._playback = []
            elif self.conf.render_mode == "plot":  # detailed plot of load and crane movement
                self.show_plot(self.nresets)

        _ = super().reset(seed=seed, options=options)

        self.nresets += 1
        if self.conf.start_speed == 0.0:  # run in 'start' mode, learning how to start the pendulum action
            assert self.wire.cm_v[0] == 0.0, f"Load speed expected zero. Found {self.wire.cm_v[0]}"  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        elif self.conf.randomize_start:
            self.wire.cm_v[0] = self.np_random.uniform(-abs(self.conf.start_speed), abs(self.conf.start_speed))  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        else:
            self.wire.cm_v[0] = self.conf.start_speed  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        self.initial_speed = float(self.wire.cm_v[0])  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
        self.acc_hist = 0.0
        self.obs, self.reward, self.trunc = self._get_obs()
        self.term = False
        if self.conf.render_mode == "play-back":
            self._append_playback(0.0)
        self.steps = 0
        self.time = 0.0
        info = self._get_info(self.reward, self.steps)
        self.rewards = [float(self.reward)]
        return self.obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
        """Advance the environment by one time step.

        Args:
            action (int): Action index selecting the crane acceleration.

        Returns:
        -------
            tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
                (observation, reward, terminated, truncated, info)
        """
        if self.conf.continuous_actions:
            acc = float(np.asarray(action).flat[0]) * self.conf.acc
        else:
            action_idx = int(action)
            if action_idx not in self.action_to_acc:
                raise KeyError(f"The action {action} is not in space {self.action_to_acc}") from None
            acc = self.action_to_acc[action_idx]
        self.crane.d_velocity[0] = acc
        self.steps += 1
        _ = self.crane.do_step(self.time, self.conf.dt)
        self.time += self.conf.dt

        obs, self.reward, self.trunc = self._get_obs(acc)
        if self.trunc and self.reward_fac.terminal_penalty != 0.0:
            self.reward += self.reward_fac.terminal_penalty
        self.rewards.append(float(self.reward))

        if self.conf.render_mode == "play-back":
            self._append_playback(self.steps)
        elif self.conf.render_mode == "reward-tracking":
            _ = self._reward_point.set_data(list(range(len(self.rewards))), self.rewards)
            _ = plt.xlim((0, len(self.rewards)))
            _ = plt.ylim((min(self.rewards), max(self.rewards)))
            plt.pause(1e-10)
        self.term = self.conf.reward_limit is not None and self.reward > self.conf.reward_limit
        if self.term:
            self.nsuccess += 1
        info = self._get_info(self.reward, self.steps)
        if self.trunc > 0:
            info["crash"] = True
        return obs, self.reward, self.term, (self.trunc > 0), info

    def render(self, save_path: str | None = None) -> None:
        """Render the current episode.

        Parameters
        ----------
        save_path : str or None, optional
            If set and render_mode is ``"plot"``, save the figure to this path
            instead of showing it interactively (default None).
        """
        if self.conf.render_mode == "play-back":
            self.show_animation()
        elif self.conf.render_mode == "plot":
            self.show_plot(self.nresets, save_path=save_path)

    def set_state(
        self,
        pos: np.ndarray | float,
        speed: np.ndarray | float,
        direction: np.ndarray | float,
        w_speed: np.ndarray | float,
    ) -> tuple[np.ndarray | tuple[int, ...], float, int]:
        """Set the state of the pendulum. Used for test purposes.

        Args:
            pos: crane position as vector or only x component
            speed: crane speed as vector or only x component
            direction: wire direction vector or angle in x-z-plane in radians
            w_speed: load speed vector or x-value of speed

        Returns:
            ``(observation, reward, truncate_flag)`` as ``self._get_obs()``
        """
        self.reset_crane()
        self.crane.position = pos if isinstance(pos, np.ndarray) else np.array((pos, 0, 0), float)
        self.crane.velocity = speed if isinstance(speed, np.ndarray) else np.array((speed, 0, 0), float)
        self.crane.d_velocity = np.array((0, 0, 0), float)
        self.crane.boom0.update_child()
        self.wire.origin_v = self.crane.velocity.copy()
        self.wire.origin_acc = np.array((0, 0, 0), float)
        self.wire.direction = (
            direction
            if isinstance(direction, np.ndarray)
            else np.array((np.sin(direction), 0, -np.cos(direction)), float)
        )
        self.wire.boom[1:] = cartesian_to_spherical(self.wire.direction)[1:]
        self.wire._c_m = self.wire.origin + self.wire.direction * self.wire.length  # noqa: SLF001
        self.wire.cm_acc = np.array((0, 0, 0), float)
        if not isinstance(w_speed, np.ndarray) and float(w_speed) > EPS:
            z_fac = -self.wire.direction[0] / self.wire.direction[2]  # ensure orthogonality of speed to direction
            self.wire.cm_v = w_speed * np.array((1, 0, z_fac), float) if isinstance(w_speed, float) else w_speed
        else:
            if isinstance(w_speed, np.ndarray) and np.dot(w_speed, self.wire.direction) > EPS:
                raise ValueError(f"Speed {w_speed} is not orthogonal to the wire {self.wire.direction}") from None
            self.wire.cm_v = w_speed if isinstance(w_speed, np.ndarray) else np.array((w_speed, 0, 0), float)
        return self._get_obs()

    def get_state(self, *, as_x: bool = True) -> tuple[float, ...]:
        """Get the current state of the pendulum.

        Args:
            as_x: Get only the x-components of the wire state (x-position and x-speed).
               Otherwise angle wrt. z and speed orthogonal to wire.

        Returns:
            a tuple of all state variables (crane-x, crane-v_x, wire-x/angle, wire-v_x/wire-|v|)
        """
        if as_x:
            return (
                self.crane.position[0],
                self.crane.velocity[0],
                self.wire.length * self.wire.direction[0],
                self.wire.cm_v[0],
            )
        v = np.sqrt(np.dot(self.wire.cm_v, self.wire.cm_v))
        return (
            self.crane.position[0],
            self.crane.velocity[0],
            np.pi - np.arctan2(self.wire.direction[0], self.wire.direction[2]),
            v if self.wire.cm_v[0] >= 0 else -v,
        )

    def analyse_observation(  # noqa: C901
        self,
        obs: tuple[int, ...],
        samples: int = 100,
        plot: tuple[tuple[int, int], ...] = ((0, 1), (2, 3)),
        *,
        show: bool = False,
    ) -> None:
        """Analyse a given discrete obesrvation tuple with respect to rewards that can be gained.

        Args:
            obs: discrete observation tuple to be analysed.
               If a value is set to -1, the respective observation element can span the whole range
            samples: number of samples to run in the analysis
            plot: tuples of two indices for which to plot reward. E.g. Phase angles
            show: optional show analysis results
        """

        def min_max(i: int, o: int) -> tuple[float, float]:
            """Get the min and max values from the discretization."""
            key = list(self.discrete.keys())[i]
            if o == -1:
                return self.discrete[key][0], self.discrete[key][-1]
            if o == len(self.discrete[key]):
                return self.discrete[key][o], self.discrete[key][o]  # single value at right end
            return (self.discrete[key][o], self.discrete[key][o + 1])

        def col(a: int) -> str:
            return ["blue", "green", "red"][a]

        acc_reward: list[float] = [0.0, 0.0, 0.0]
        data: list[list[float]] = [[] for _ in range(len(plot) * 3 * 3)]  # plot*action*3D
        info = "Analysis of "

        for n in range(samples):
            state: list[float] = []
            for i, o in enumerate(obs):
                key = list(self.discrete.keys())[i]
                mm = min_max(i, o)
                state.append(self.np_random.uniform(*mm))
                if n == 0:
                    if i == 0:
                        info += f"{key}:{np.degrees(mm[0]):2.3f}-{np.degrees(mm[1]):2.3f}, "
                    else:
                        info += f"{key}:{mm[0]:2.3f}-{mm[1]:2.3f}, "
            for a in range(3):
                acc = (a - 1) * self.conf.acc
                self.set_state(
                    pos=self.wire.length * np.sin(state[0]), speed=state[1], direction=state[2], w_speed=state[3]
                )
                _, _r0, _ = self._get_obs(0.0)
                self.step(a)
                _obs, _r, _t = self._get_obs(acc)
                acc_reward[a] += _r - _r0

                i = 0
                for k in range(len(plot)):
                    data[i].append(state[plot[k][0]])
                    data[i + 1].append(state[plot[k][1]])
                    data[i + 2].append(_r - _r0)
                    i += 3

        LOGGER.info(f"Rewards: {[acc_reward[a] / samples for a in range(3)]}")
        if len(plot) and show:
            ax: list[int] = []
            fig = plt.figure()  # type: ignore[reportAttributeAccessIssue]
            idx = 0
            for k in range(len(plot)):
                for a in range(3):
                    ax.append(fig.add_subplot(3, 2, idx + 1, projection="3d"))
                    ax[idx].scatter(  # type: ignore[reportAttributeAccessIssue]
                        *data[a * len(plot) * 3 + k * 3 : a * len(plot) * 3 + (k + 1) * 3], c=col(a), marker="."
                    )
                    idx += 1
            plt.show()

    def get_parameters(self) -> dict[str, Any]:
        """Return the environment parameter settings as dict."""
        info = asdict(self.conf)
        if "discretization" in info and self.conf.discrete != "none":
            if info["discretization"] is None:
                info["discretization"] = self.discrete
            vals = info["discretization"]
            _vals: dict[str, list[float]] = {}
            for k, v in vals.items():
                _vals.update({k: [float(_v) for _v in v]})  # make it serializable - list
            info["discretization"] = _vals
        return info

    def reward_stats_calc(self, steps: int) -> None:
        """After an episode is run, analyse the .rewards list statistically.

        * number of steps for the episode
        * average reward gain over episode
        * standard deviation of reward gains
        * reward gain trend over episode
        The list is then reset before the next episode is run.

        Args:
            steps (int): number of steps in this episode.
        """
        rewards = np.array(self.rewards, float)
        self.reward_stats["steps"].append(steps)  # number of steps in episode
        self.reward_stats["reward"].append(float(rewards[-1]))  # final reward
        self.reward_stats["status"].append(-1 if self.trunc else (1 if self.term else 0))
        m, c = np.linalg.lstsq(np.stack([np.arange(steps), np.ones(steps)]).T, np.log(np.abs(rewards)))[0]
        self.reward_stats["start"].append(-np.exp(c))  # start reward
        self.reward_stats["relaxation"].append(-1 / m)  # reward relaxation time (exponential model)


def _level(val: float, categories: tuple[float, ...]) -> int:
    """Determine the bucket index for a value given ordered categories.

    val < categories[0] => -1, categories[k] <= val < categories[k+1] => k, val>=categories[-1] => -1

    Args:
        val (float): Value to classify.
        categories (tuple[float, ...]): Ordered category boundaries.

    Returns:
    -------
        tuple[int, int]: ``bucket_index`` of value with respect to categories. -1 if outside categories.
    """
    for i, x in enumerate(categories):
        if val < x:
            return i - 1
    return -1
