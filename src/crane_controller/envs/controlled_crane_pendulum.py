"""Gymnasium environment for the crane anti-pendulum task."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from component_model.utils.transform import cartesian_to_spherical
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


def _level(val: float, categories: tuple[float, ...]) -> int:
    """Determine the bucket index for a value given ordered categories.

    val < categories[0] => -1, categories[k] <= val < categories[k+1] => k, val>=categories[-1] => -1

    Args:
        val (float): Value to classify.
        categories (tuple[float, ...]): Ordered category boundaries.

    Returns:
        tuple[int, int]: ``bucket_index`` of value with respect to categories. -1 if outside categories.
    """
    for i,x in enumerate(categories):
        if val < x:
            return i-1
    return -1

# Observation is either a discrete tuple or a continuous ndarray
AntiPendulumObs = tuple[int, ...] | np.ndarray


class AntiPendulumEnv(gym.Env[AntiPendulumObs, int | np.ndarray]):
    """Environment for a py-crane-based anti-pendulum task.

    Uses the matplotlib-based animation module from py-crane.

    Args:
        crane (Callable[..., Crane]): Factory callable that creates the crane object.
        acc (float)=0.1: Acceleration magnitude applied to the crane.
        start_speed (float)=1.0: Fixed start speed in m/s. A negative value causes a random speed
           in the range ``[-|start_speed|, |start_speed|]`` each episode
        render_mode (str)='none': One of the modes listed in ``metadata["render_modes"]``
        size (float)=0.0: Axis length in all directions
        rail_limit (float): Half-span of the crane rail in metres (default 10.0). The crane spans
            ``+-rail_limit``; within PPO an episode is truncated when ``|x| > rail_limit``.
        seed (int)=None: Seed for repeatable random numbers.
        reward_limit (float)=None: Reward at which an episode is terminated and the anti-pendulum is deemed successful
        reward_truncate (float)=None: Reward at which an episode is truncated.
           Environment sets this reward to signal truncation. 
        dt (float)=1.0: Simulation time step
        discrete (dict[str, tuple[float | int, ...]]: When provided, activates discrete observation mode with the given
           category boundaries. Expected keys: `angle`,`pos`,`speed`,`distance`,`crane-sector`,`crane-speed` 
        reward_fac (tuple[float,...])=(-1.0,-1.0,-0.5): Weights between reward contributions
        discount (float) = 0.8: discount factor for acceleration history to include in (discrete) observation
        continuous_actions (bool)=False: If True, the action space is ``Box([-1], [1])`` and an action value
            in ``[-1, 1]`` is scaled by ``acc`` to produce the crane acceleration.
            If False, the action space is ``Discrete(3)`` with mapping``0=-acc, 1=0, 2=+acc`` (Q-agent compatible).
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

    DEFAULT_DISCRETE: ClassVar[dict[str, tuple[float | int, ...]]] = {
        "angle": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
        "distance": (0.0, 0.5, 1.0, 2.0),
        "pos": (0, 1),
        "speed": (0, 1),
        "c_pos": (0, 1),
        "c_speed": (0, 1),
        'avg-acc': np.linspace(-1.25, 1.25, 11),
    }
    DISCRETE2: ClassVar[dict[str, tuple[float | int, ...]]] = {
        'angle': np.radians((-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32)),
        'speed': np.linspace(-5,5,11), # only x-component to preserve sign!
        'c-pos': np.array( (-2.0, -1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0, 2.0), float),
        'c-speed': np.array( (-2.0, -1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0, 2.0), float),
        'avg-acc': np.linspace(-1.25, 1.25, 11),
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
        reward_limit: float|None = None,
        reward_truncate: float|None = None,
        dt: float = 1.0,
        discrete: dict[str, tuple[float | int, ...]] | None = None,
        reward_fac: RewardConfig | None = None,
        continuous_actions: bool = False,  # noqa: FBT001, FBT002
        discount: float = 0.8,
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
        self.discount = discount
        self.reward_stats: list[list[float]] = []
        self._playback: list[list[float]] = []
        self.rewards: list[float] = []
        if render_mode == "reward-tracking":
            self._reward_point = self._reward_plot_init()
        elif render_mode == "plot":
            self.traces: dict[str, list[float]] = {"c_x": [], "c_v": [], "l_x": [], "l_v": [], "acc": []}

        self.obeservation_space: spaces.Box | spaces.Discrete  # pyright: ignore[reportMissingTypeArgument]  # Discrete type arg not needed here
        # Continuous observations are crane position, crane velocity, wire polar angle, and load x-velocity.
        max_speed = np.sqrt(9.81 * self.wire.length)  # speed for pendulum at +/- 90 deg. Polar as deflection from -z
        self.discrete : dict[str, tuple[float | int, ...]] = {} # set by .init_discrete + observation_space
        self.acc_hist : float = 0.0 # used for DISCRETE2. Set by .init_discrete

        if discrete is not None:
            self.init_discrete(discrete)
        else:
            self.discrete = {}
            self.spaces_min = np.array((-rail_limit, -max_speed, 0.0, -max_speed), float)
            self.spaces_max = np.array((rail_limit, max_speed, np.pi, max_speed), float)
            self.observation_space = spaces.Box(self.spaces_min, self.spaces_max, shape=(4,), dtype=np.float64)

        self.dt = dt
        self.acc = acc
        #self.dist_d2_max = abs(self.distance_max) + abs(self.speed_max*self.dt)
        self.tau_max = self.distance_max / self.acc/ self.dt # time with min. speed from 0 to end

        self.nresets: int = 0
        self.start_speed = start_speed
        _ = super().reset(seed=seed)
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
        self.obs : tuple[int, ...] | np.ndarray # previous observation
        self.energy0 : float = 0.0 # save the initial energy (set by reset())

    def init_discrete(
        self,
        spec: dict[str, tuple[float | int, ...]] | None = None,
    ) -> None:
        """Translate the observation-space spec into a MultiDiscrete space.

        Expected keys in default spec::

            'angle'    - amplitude categories (converted to energy levels)
            'distance' - distance categories from origin
            'pos'      - load position (+/- x)
            'speed'    - load speed (+/- x)
            'c_pos'    - crane position sector (+/- x)
            'c_speed'  - crane speed (+/- x)
            'avg-acc'  - average acceleration history

        Args:
            spec (dict[str, tuple[float | int, ...]]): Mapping of observation dimension names to category boundaries.

        Returns:
            tuple[spaces.MultiDiscrete, dict[str, tuple[float | int, ...]]]: The constructed ``MultiDiscrete`` space
            and the updated spec (with ``'angle'`` replaced by ``'energy'``).
        """
        _spec = spec.copy() if spec is not None else AntiPendulumEnv.DEFAULT_DISCRETE.copy()
        self.acc_hist = 0.0
        if 'distance' in _spec:
            # We replace the angle with pendulum energy levels, which are easier to use for observation calculation
            angle = _spec.pop("angle")
            energy = [9.81 * self.wire.length * (1.0 - np.cos(np.radians(a))) for a in angle]
            spec_e = {"energy": tuple(energy)}
            
            for k, v in _spec.items():
                spec_e.update({k: v})
    
        elif 'speed' in _spec: #DISCRETE2
            spec_e = _spec
        else:
            raise ValueError("Unknown discretization {_spec}") from None

        self.observation_space = spaces.MultiDiscrete(np.array([len(spec_e[k]) for k in spec_e]))
        self.discrete = spec_e
    
    @property
    def energy_max(self):
        try:
            return self.discrete['energy'][-1]
        except KeyError as err1:
            try:
                return 0.5*self.discrete['speed'][-1]**2
            except KeyError as err2:
                logger.error(f"'energy' or 'speedæ not part of discretization, => maximum value not defined: {err2}")
    
    @property
    def distance_max(self):
        try:
            return self.discrete['distance'][-1]
        except KeyError as err1:
            try:
                return self.discrete['c-pos'][-1]
            except KeyError as err2:
                logger.error(f"'distance' or 'c-pos' not part of discretization. => maximum value not defined: {err2}")
                        
    @property
    def speed_max(self):
        try:
            return self.distance_max/ self.dt/ 10
        except KeyError as err:
            logger.error(f"'distance' not part of discretization. => maximum speed value is not defined: {err}")

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

    def show_plot(self, episode: int, save_path: str | None = None) -> None:
        """Plot detailed traces for a single episode.

        Args:
            episode (int): Episode number used in the plot title.
            save_path (str)=None: If set, save the figure to this path and close it instead of calling ``plt.show()``
        """
        if not self.traces["l_v"]:
            return
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
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
        ax6.set_xlabel("time [s]")
        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
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

    def _get_continuous_obs(self) -> tuple[np.ndarray, int]:
        """Return continuous observations and an out-of-bounds error flag.

        Returns:
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

    def _get_discrete_obs(self, energy: float, acc:float) -> tuple(tuple[int, ...],bool):
        """Return the discrete observation tuple from the current crane state.

        Args:
            energy (float):  Current pendulum energy.
            acc (float): current acceleration command

        Returns:
            tuple[int, ...]: Discretised observation ``(energy_level, side, speed_sign, distance_level, sector)``.
        """
        self.acc_hist = self.discount* self.acc_hist + (1.0-self.discount)* acc
        if 'distance' in self.discrete:
            obs = [_level(energy, self.discrete["energy"]),
                   _level(abs(self.crane.position[0]), self.discrete["distance"]),
                   int(self.wire.end[0] - self.wire.origin[0] < 0.0),
                   int(self.wire.cm_v[0] < 0.0),  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
                   int(self.crane.position[0] < 0.0),
                   int(self.crane.velocity[0] < 0.0),
                   _level(self.acc_hist, self.discrete['avg_acc']),
                ]
        elif 'speed' in self.discrete: 
            angle = np.pi - self.wire.boom[1]
            der_angle = np.arctan2( self.wire.cm_v[0], self.wire.length - self.wire.cm_v[2])
            obs = [_level(np.pi - self.wire.boom[1], self.discrete['angle']),
                   _level(self.wire.cm_v[0], self.discrete['speed']), # only x-component, to keep sign!
                   _level(self.crane.position[0], self.discrete['c-pos']),
                   _level(self.crane.velocity[0], self.discrete['c-speed']),
                   _level(self.acc_hist, self.discrete['avg-acc']),
                   ]
        else:
            raise ValueError(f"Unknown discretization {self.discrete}.") from None
        trunc = any(i<0 for i in obs)
        return (tuple(obs), trunc)
        

    def _get_obs(self, acc: float = 0.0) -> tuple[np.ndarray | tuple[int, ...], float, int]:
        """Compute the current observation, the reward and the truncation flag from the crane state.

        In discrete mode the observation keys are as defined in .DEFAULT_DISCRETE
        
        Args:
            acc (float): Acceleration used to get to this state (for use in traces)

        Returns:
            tuple[np.ndarray | tuple[int, ...], float, int]: ``(observation, reward, truncate_flag)``.
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

        if len(self.discrete):
            self.obs, truncate = self._get_discrete_obs(energy, acc)
        else:
            self.obs, truncate = self._get_continuous_obs()

        if self.render_mode == "plot":
            self.traces["c_x"].append(self.crane.position[0])
            self.traces["c_v"].append(self.crane.velocity[0])
            self.traces["l_x"].append(self.wire.c_m[0])
            self.traces["l_v"].append(self.wire.cm_v[0])  # pyright: ignore[reportUnknownMemberType]  # dynamic attr on Wire
            self.traces["acc"].append(acc)

        return (self.obs, self.reward, truncate)

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

        Returns:
            tuple[tuple[int, ...] | np.ndarray, dict[str, float | int]]: Initial observation and info dict.
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
        self.rewards = [float(self.reward)]
        return self.obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
        """Advance the environment by one time step.

        Args:
            action (int): Action index selecting the crane acceleration.

        Returns:
            tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
                (observation, reward, terminated, truncated, info)
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
            self.rewards.append(float(self.reward))

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


    def set_state(self, pos:np.ndarray|float, speed:np.ndarray|float, direction:np.ndarray|float, w_speed:np.ndarray|float):
        """Set the state of the pendulum. Used for test purposes.

        Args:
            pos (ndarray|float): crane position as vector or only x component
            speed (ndarray|float): crane speed as vector or only x component
            direction (ndarray|float): wire direction vector or polar angle in radians
            w_speed (ndarray|float): load speed vector or x-value of speed
        """
        self.crane.position = np.array((pos,0,0), float) if isinstance(pos, float) else pos
        self.crane.velocity = np.array((speed,0,0), float) if isinstance(speed, float) else speed
        self.crane.d_velocity = np.array((0,0,0),float)
        self.crane.boom0.update_child()
        self.wire.origin_v = self.crane.velocity
        self.wire.origin_acc = np.array((0,0,0), float)
        self.wire.direction = np.array((np.sin(direction),0,-np.cos(direction)), float) if isinstance(direction, float) else direction
        self.wire.boom[1:] = cartesian_to_spherical(self.wire.direction)[1:]
        self.wire._c_m = self.wire.origin + self.wire.direction* self.wire.length
        self.wire.cm_v = np.array((w_speed,0,0), float) if isinstance(w_speed, float) else speed
        self.wire.cm_acc = np.array((0,0,0), float)
        #self.wire.pendulum_relax()
        if not isinstance(w_speed, float) or float(w_speed) > 1e-10:
            z_fac = -self.wire.direction[0]/self.wire.direction[2] # ensure orthogonality of speed to direction
            self.wire.cm_v = w_speed* np.array((1,0,z_fac), float) if isinstance(w_speed, float) else w_speed
        self.wire.calc_statics_dynamics(None)

        

    def get_parameters(self) -> dict[str, Any]:
        """Return the environment parameter settings as dict."""
        return {
            "wire-length": self.wire.length,
            "wire-q-factor": self.wire.q_factor,
            "reward-factors": self.reward_fac,
            "acceleration": self.acc,
            "step-size": self.dt,
            "observations-discretization": None if not hasattr(self, "discrete") else self.discrete,
            "reward_limit": self.reward_limit,
            "start-load-speed": self.start_speed,
        }

    def reward_stats_calc(self, steps:int):
        """After an episode is run, analyse the .rewards list statistically.
        The list is then reset before the next episode is run.

        * number of steps for the episode
        * average reward gain over episode
        * standard deviation of reward gains
        * reward gain trend over episode
        """
        rewards = np.array(self.rewards, float)
        avg = np.average( rewards)
        std = np.std(rewards)
        avg_gain = np.average( rewards[1:] - rewards[:-1])
        std_gain = np.std( rewards[1:] - rewards[:-1])
        gain_trend = np.average( rewards[2:] - 2* rewards[1:-1] + rewards[:-2])
        return (steps, avg, std, avg_gain, std_gain, gain_trend)
        
        
