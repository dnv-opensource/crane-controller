import logging
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from py_crane.animation import AnimatePlayBackLines

logger = logging.getLogger(__name__)


class AntiPendulumEnv(gym.Env):
    """Environment of the controlled py-crane based mobile crane,
    using the matplotlib-based animation module from py-crane.
    Simple case of anti-pendulum application.

    Args:
        crane (py_crane): the crane object to use as basis
        acc (float): the acceleration (-,0,+) used on the crane to perform anti-pendulum actions
        start_speed (float) = 1.0: Fixed start angle (in degrees) or
          if <0 random at every episode with [-start_speed ... start_speed]
        render_modes (str): see metadata dict
        size (int): The axis length in all directions, but -z
        seed (int): Optional possibility to use repeatable random numbers. None: not repeatable.
        reward_limit (float) = 1e-3: reward at which episodes are terminated and anti-pendulum is deemed successful.
        dt (flaot) = 0.01: time steps for the simulation
        discrete (dict): either empty dict (continuous mode) or description of discrete ranges:
           'angles': categories of angles for the amplitude of the load,
           'pos': current position of the load (+/- x),
           'speed': current speed of the load (+/- x-direction)
           'distance': distance categories from origin,
           'sector': current sector of the crane position (+/- x)
    """

    metadata = {
        "render_modes": [
            "none",
            "play-back",
            "data",
            "reward-tracking",
            "plot",
            "agent",
        ],
        "interval": 100,
        "show-len-1": False,
        "x-max": None,
    }

    def __init__(
        self,
        crane: Any,
        acc: float = 0.1,
        start_speed: float = 1.0,
        render_mode: str = "none",
        size: float = 10.0,
        seed: int | None = None,
        reward_limit: float = 50.0,
        dt: float = 0.01,  # time steps for the simulation
        discrete: dict[str, tuple[float, ...]] | None = None,
    ):
        self.crane_maker = crane
        self.crane = crane()
        self.wire = self.crane.boom_by_name("wire")
        assert self.wire is not None, "Need a crane wire!"
        assert render_mode in self.metadata["render_modes"], f"render_mode: {render_mode}"
        self.render_mode = render_mode
        self.reward_stats: list[list[float]] = []
        self._playback: list[list[float]] = []
        self.rewards: list[float] = []
        if render_mode == "reward-tracking":
            self._reward_point = self._reward_plot_init()
        elif render_mode == "plot":
            self.traces: dict[str, list[float]] = {"c_x": [], "c_v": [], "l_x": [], "l_v": []}

        self.obeservation_space: spaces.Box | spaces.Discrete
        # Observations is a 4-dim np-array with
        # (crane-x, crane-v_x, load-polar-angle_x, load-v_x)
        self.min_speed = 0.1  # np.sqrt(2*reward_limit) # starting with less does not make sense (goal already reached)
        max_speed = np.sqrt(9.81 * self.wire.length)  # speed for pendulum at +/- 90 deg. Polar as deflection from -z
        if discrete is not None:
            self.observation_space, self.discrete = self._init_discrete(discrete)
        else:
            self.discrete = {}
            self.spaces_min = np.array((-size, -max_speed, -np.pi / 2, -max_speed), float)
            self.spaces_max = np.array((size, max_speed, np.pi / 2, max_speed), float)
            self.observation_space = spaces.Box(self.spaces_min, self.spaces_max, shape=(4,), dtype=np.int64)

        self.nresets: int = 0
        # self.reset(seed)
        self.acc = acc
        self.start_speed = start_speed
        self.size = size
        self.figsize: tuple[float, float] = (-size, size)  # The matplotlib animation window
        self.reward_limit = reward_limit
        self.nsuccess: int = 0
        self.reward = 0.0  # a basic reward (pendulum energy + distance measure)
        self.dt = dt

        # We have 1 acceleration action which can each be min, zero or max, corresponding to acceleration of crane
        self.action_space = spaces.Discrete(3, start=0, seed=42, dtype=np.int64)
        self.action_to_acc = {0: -self.acc, 1: 0.0, 2: self.acc}

    def _init_discrete(self, spec: dict[str, tuple[float, ...]]):
        """Translate the spec of the observation space to a useful MultiDiscrete Space object.

        The following keys are expected
           'angles': categories of angles for the amplitude of the load,
           'pos': current position of the load (+/- x)
           'speed': current speed of the load (+/- x-direction),
           'distance': distance categories from origin,
           'sector': current sector of the crane position (+/- x)
        """
        # We replace the angles with pendulum energy levels, which are easier to use for observation calculation
        observation_space = spaces.MultiDiscrete(np.array([len(spec[k]) for k in spec]))
        angles = spec.pop("angles")
        energies = [9.81 * self.wire.length * (1.0 - np.cos(np.radians(a))) for a in angles]
        spec.update({"energies": tuple(energies)})
        return (observation_space, spec)

    def _reward_plot_init(self, marker: str = ""):
        if not len(marker):
            point = plt.plot(0, 0)[0]
        else:
            point = plt.plot(0, 0, marker)[0]
        plt.ion()
        plt.show()
        return point

    def _append_playback(self, time: float):
        """Append the current state of the crane to ._playback."""
        if not len(self._playback):  # no records there yet
            self._playback.append([time])  # slot 0 for time
            for b in self.crane.booms():
                self._playback.append([b.end])  # one slot per boom (first boom is fixation)
        else:
            self._playback[0].append(time)
            for i, b in enumerate(self.crane.booms()):
                self._playback[i + 1].append(b.end)  # add vector to the correct slot.

    def show_animation(self):
        """Show the playback animation of the episode (the current recording of ._playback."""
        if len(self._playback[0]) < 2 and not AntiPendulumEnv.metadata["show-len-1"]:  # nothing really to show
            return
        data: list[np.ndarray] = []
        for col in self._playback:
            data.append(np.array(col, float))  # make sure that we have an ndarray
        ani = AnimatePlayBackLines(
            data=data,
            lw=(5, 1),
            figsize=(10, 10),
            interval=AntiPendulumEnv.metadata["interval"],
            title="Anti-Pendulum episode",
        )
        ani.do_animation()

    def show_plot(self, episode):
        """Plot the report_mode='plot' figure."""
        fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
        times = np.arange(len(self.traces["c_x"]))
        damping = self.traces["l_v"][0] * np.exp(-times / self.wire.damping_time)
        ax1.plot(times, self.traces["l_x"], label="load angle", color="blue")
        ax1y2 = ax1.twinx()
        ax1y2.plot(times, self.traces["l_v"], label="load speed", color="red")
        ax1y2.plot(times, damping, label="damping", color="green")
        ax2.plot(times, self.traces["c_x"], label="crane pos", color="blue")
        ax2y2 = ax2.twinx()
        ax2y2.plot(times, self.traces["c_v"], label="crane speed", color="red")
        ax3.plot(list(range(len(self.rewards))), self.rewards, label="rewards")
        ax1.legend()
        ax2.legend()
        plt.title(f"Detailed plot of episode {episode}, reward:{self.reward}")
        plt.show()
        for k in self.traces.keys():
            self.traces[k] = []
        self.rewards = []

    def _get_obs(self) -> tuple[np.ndarray | tuple, float, int]:
        """Translate the current crane state into an observation useable by gym and calculate the related reward.

        For discrete mode:
           'energies': categories of energies of the load,
           'side': current side of the load (+/- x),
           'distance': distance categories from origin,
           'sector': current sector of the crane position (+/- x)
        """
        err = 0

        def level(idx: int, val: float, categories: tuple[float, ...]):
            """Determine the level with respect to given categories.
            If val<categories[0] or val>categories[-1] and error is flagged (leads to termination).
            """
            if val < categories[0]:
                _err = idx
                return 0
            for i, c in enumerate(categories):
                if val <= c:
                    return i
            _err = idx
            return len(categories) - 1

        energy = 9.81 * self.wire.end[2] + 0.5 * np.dot(self.wire.cm_v, self.wire.cm_v)
        if self.start_speed == 0.0:  # start pendulum mode
            reward = energy
        else:  # stop pendulum mode
            reward = -energy
            if np.sign(self.crane.position[0]) == np.sign(self.crane.velocity[0]):  # moving away from origo
                reward -= (
                    0.0015 * self.wire.length * (abs(self.crane.position[0]) + self.crane.velocity[0] ** 2 / self.acc)
                )
                # if the crane moves towards the origo we do not add 'energy'
        self.reward = reward

        obs: tuple[int, ...] | np.ndarray
        if len(self.discrete):
            obs = (
                level(1, energy, self.discrete["energies"]),  # energy level
                int(self.wire.end[0] - self.wire.origin[0] < 0.0),  # load position >=0 / < 0 in x-direction
                int(self.wire.cm_v[0] < 0.0),  # load speed >=0 / < 0 in x-direction
                level(3, abs(self.crane.position[0]), self.discrete["distance"]),  # distance from origin
                int(self.crane.position[0] < 0.0),  # sector with respect to origin, where the crane is
            )

        else:
            # polar measured downwards as +/- deflection from the negative z-axis
            polar = np.pi - self.wire.boom[1] if abs(self.wire.boom[2]) < 0.1 else -(np.pi - self.wire.boom[1])
            obs = np.array(
                (
                    self.crane.position[0],
                    self.crane.velocity[0],
                    polar,
                    self.wire.cm_v[0],
                ),
                float,
            )

            for i, (_obs, _min, _max) in enumerate(zip(obs, self.spaces_min, self.spaces_max, strict=True)):
                if not _min <= _obs <= _max:
                    err = i + 1

        if self.render_mode == "plot":
            self.traces["c_x"].append(self.crane.position[0])
            self.traces["c_v"].append(self.crane.velocity[0])
            self.traces["l_x"].append(self.wire.c_m[0])
            self.traces["l_v"].append(self.wire.cm_v[0])

        return (obs, reward, err)

    def low_reward(self):
        if self.start_speed == 0.0:
            return 0.0
        return -self.discrete["energies"][-1]

    def _get_info(self, reward: float, steps: int):
        return {"steps": steps, "reward": reward}

    def reset_crane(self):
        self.crane.position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.d_velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.crane.torque = np.array((0, 0, 0), dtype=np.float64)
        self.crane.force = np.array((0, 0, 0), dtype=np.float64)
        self.crane.current_time = 0.0  # used to make current_time from do_step known for the whole crane
        self.crane.boom0.update_child()
        self.crane.calc_statics_dynamics(None)
        self.wire.pendulum_relax()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the crane for a new episode."""
        self.reset_crane()

        if self.nresets <= 0:  # reset during instantiation. Initialize
            if self.render_mode == "data":
                self._reward_point = self._reward_plot_init("b.")

        else:  # reset between episodes. Data are available
            self.reward_stats.append([self.steps, self.reward])
            if self.render_mode == "data":
                self._reward_point.set_data([r[0] for r in self.reward_stats], [r[1] for r in self.reward_stats])
                plt.pause(1e-10)
            elif self.render_mode == "play-back" and len(self._playback):
                self.show_animation()
                self._playback = []
            elif self.render_mode == "plot":
                self.show_plot(self.nresets)

        super().reset(seed=seed, options=options)

        self.nresets += 1
        if self.start_speed == 0.0:  # run in 'start' mode, learning how to start the pendulum action
            assert self.wire.cm_v[0] == 0.0, f"Load speed expected zero. Found {self.wire.cm_v[0]}"
        elif self.start_speed < 0.0:  # random speed in 'stop' mode [-,+] range
            speed = self.np_random.uniform(
                -(-self.start_speed - self.min_speed),
                (-self.start_speed - self.min_speed),
            )
            self.wire.cm_v[0] = speed + self.min_speed if speed >= 0 else speed - self.min_speed
        else:  # fixed speed in 'stop' mode (more control)
            self.wire.cm_v[0] = self.start_speed
        obs, self.reward, _ = self._get_obs()
        if self.render_mode == "play-back":
            self._append_playback(0.0)
        self.steps: int = 0
        info = self._get_info(self.reward, self.steps)
        return obs, info

    def step(self, action):
        """Step in the environment according to the given action."""
        self.crane.d_velocity[0] = self.action_to_acc[int(action)]
        self.steps += 1
        self.crane.do_step(self.steps, self.dt)

        obs, self.reward, truncated = self._get_obs()
        # print(f"@{self.steps}({action-1}) => {self.reward:4.3f}. Crane:{obs[0]:4.3f},{obs[1]:4.3f}. load:{np.degrees(obs[2]):4.3f},{obs[3]:4.3f}")
        if self.render_mode != "none":
            self.rewards.append(self.reward)

        if self.render_mode == "play-back":
            self._append_playback(self.steps)
        elif self.render_mode == "reward-tracking":
            self._reward_point.set_data(list(range(len(self.rewards))), self.rewards)
            plt.xlim((0, len(self.rewards)))
            plt.ylim((min(self.rewards), max(self.rewards)))
            plt.pause(1e-10)
        terminated = self.reward > self.reward_limit
        if terminated:
            self.nsuccess += 1
        info = self._get_info(self.reward, self.steps)
        # if terminated or truncated>0:
        #     print(f"RESET@{self.steps}. {self.reward}, {self.reward}. Reason:{'terminated' if terminated else truncated}. Obs:{obs}")
        return obs, self.reward, terminated, (truncated > 0), info

    def render(self):
        """Render a frame."""
        if self.render_mode == "play-back":  # show the animation
            self.show_animation()
