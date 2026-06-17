"""A simple environment for general Q-Lerning tests."""

import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)  # frozen=True, slots=True,
class Config:
    """Configuration parameters for SimpleTestEnv.

    Args:
        pos_range (tuple[int,int])=(-100,100): The range of positions (quantified as int)
        speed_range (tuple[int,int])=(-10,10): The range of speeds (quantified as int)
        acc (float)=1.0: The fixed acceleration (-acc, 0.0, +acc)
        pos0  (float)=0.0: default initial position. Can be changed by initialize()
        speed0 (float)=0.0: default initial speed. Can be changed by initialize()
        pos1 (float)=10.0: default position goal. Can be changed by initialize()
        speed1 (float)=0.0: default speed goal. Can be changed by initialize()
        seed (int): random seed value
    """

    pos_range: tuple[int, int] = (-100, 100)
    speed_range: tuple[int, int] = (-10, 10)
    acc: float = 1.0
    pos0: float = 0.0
    speed0: float = 0.0
    pos1: float = 10.0
    speed1: float = 0.0
    seed: int = 43


class SimpleTestEnv(gym.Env[tuple[int, ...] | np.ndarray, int]):
    """A simple test environment for testing the Q-learning agent.

    Actions:
        Fixed acceleration (-,0,+)* acc
    Observations:
        Position, quantified as int in range pos_range
        Speed, quantified as int in range speed_range

    Starting from initial position and speed, the goal is to arrive at final position and speed as quick as possible.

    """

    def __init__(
        self,
        config: Config | None = None,
        reward_fac: tuple[float, float] = (1.0, 1.0),
        reward_limit: float | None = None,
        dt: float = 1.0,
        render_mode: str = "none",
    ) -> None:
        """Initialize the SimpleTest environment.

        Args:
            config: Optional configuration object with default environment parameters
            reward_fac: the reward factors on positional and speed-related observations
            reward_limit: The reward at which the episode is terminated (success)
            dt: Time step size
            render_mode: render mode: 'none', 'plot'
        """
        self.config = Config() if config is None else config
        _ = super().reset(seed=self.config.seed)
        self.reward_fac = reward_fac
        self.observation_space = spaces.MultiDiscrete(
            nvec=np.array([len(self.config.pos_range) + 1, len(self.config.speed_range) + 1], np.int16),
            dtype=np.int16,
            start=np.array([-self.config.pos_range[0], -self.config.speed_range[0]], np.int16),
        )
        self.action_space = spaces.Discrete(n=3, start=0, dtype=np.uint16)
        self.render_mode = render_mode
        self.steps = 0
        self.time = 0.0
        self.dt = dt  # time step setting
        self.pos = self.pos0 = self.config.pos0  # initial position
        self.speed = self.speed0 = self.config.speed0  # initial speed
        self.pos1 = self.config.pos1  # goal position
        self.speed1 = self.config.speed1  # goal speed
        self.reward_limit = reward_limit if reward_limit is not None else self.calc_reward(self.pos1, self.speed1)
        logger.info(f"Reward limit: {self.reward_limit}")
        self.rewards: list[float] = []
        self.traces: dict[str, list[float]] = {}
        self.nresets: int = 0
        self.nsuccess: int = 0

    def initialize(
        self,
        pos1: float,
        speed1: float,
        pos0: float = 0.0,
        speed0: float = 0.0,
    ) -> None:
        """Initialize the environment before a run (positions and speeds).

        Args:
            pos1: Goal position as int
            speed1: Goal speed as int
            pos0: Optional start position as int
            speed0: Optional start speed as int
        """
        self.pos = self.pos0 = pos0
        self.speed = self.speed0 = speed0
        self.pos1 = pos1
        self.speed1 = speed1

    def calc_reward(self, pos: float, speed: float) -> float:
        """Calculate the reward based on internal information, position and speed."""
        reward = self.reward_fac[0] * (2 * abs(self.pos1 - self.pos0) - abs(self.pos1 - pos)) + self.reward_fac[1] * (
            2 * abs(self.speed1 - self.speed0) - abs(self.speed1 - speed)
        )
        return reward

    def _get_obs(self, _acc: float) -> tuple[np.ndarray | tuple[int, ...], float, int]:
        """Compute the current observation, the reward and truncation flag.

        The observations are current position and speed.

        Args:
            acc (float): Acceleration (for use in traces)

        Returns:
        -------
            tuple[np.ndarray | tuple[int, ...], float, int]: ``(observation, reward, truncate_flag)``.
        """
        obs = (int(round(self.pos, 0)), int(round(self.speed, 0)))
        reward = self.calc_reward(self.pos, self.speed)
        truncate = not (
            (self.config.pos_range[0] < self.pos < self.config.pos_range[-1])
            and (self.config.speed_range[0] < self.speed < self.config.speed_range[-1])
            and reward > 0
        )
        return (obs, reward, truncate)

    def _get_info(self, reward: float, steps: int) -> dict[str, float | int]:
        return {"steps": steps, "reward": reward}

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
        -------
            tuple[tuple[int, ...] | np.ndarray, dict[str, float | int]]: Initial observation and info dict.
        """
        if self.nresets > 0 and self.render_mode == "plot":
            self.show_plot(self.nresets)
        self.nresets += 1
        self.pos = self.pos0
        self.speed = self.speed0
        _ = super().reset(seed=seed, options=options)
        obs, self.reward, _ = self._get_obs(0.0)
        if self.render_mode != "none":
            self.rewards.append(self.reward)
        self.steps = 0
        self.time = 0.0
        info = self._get_info(self.reward, self.steps)
        if self.render_mode == "plot":
            self.traces = {"t": [0.0], "x": [self.pos], "v": [self.speed], "a": [0.0]}
        return obs, info

    def step(self, action: int) -> tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
        """Advance the environment by one time step.

        Args:
            action (int): Action index [0,2], selecting the crane acceleration.

        Returns:
        -------
            tuple[tuple[int, ...] | np.ndarray, float, bool, bool, dict[str, float | int]]:
                (observation, reward, terminated, truncated, info)
        """
        acc = -self.config.acc if action == 0 else (0.0 if action == 1 else self.config.acc)
        self.steps += 1
        self.pos += self.speed * self.dt + 0.5 * acc * self.dt * self.dt
        self.speed += acc * self.dt
        self.time += self.dt
        if self.render_mode == "plot":
            self.traces["t"].append(self.time)
            self.traces["x"].append(self.pos)
            self.traces["v"].append(self.speed)
            self.traces["a"].append(acc)

        obs, self.reward, truncated = self._get_obs(acc)
        if self.render_mode != "none":
            self.rewards.append(self.reward)

        terminated = self.reward > self.reward_limit
        if terminated:
            self.nsuccess += 1
        info = self._get_info(self.reward, self.steps)
        return obs, self.reward, terminated, (truncated > 0), info

    def show_plot(self, episode: int) -> None:
        """Plot detailed traces for a single episode.

        Args:
            episode (int): Episode number used in the plot title.
        """
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
        times = self.traces["t"]
        ax1.plot(times, self.traces["x"], label="position", color="blue")
        ax1y2 = ax1.twinx()
        ax1y2.plot(times, self.traces["v"], label="speed", color="red")
        ax2.plot(times, self.traces["a"], label="x-acceleration", color="green")
        ax3.plot(times, self.rewards, label="rewards")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1y2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        _ = ax2.legend()
        _ = ax3.legend()
        _ = plt.suptitle(f"Detailed plot of episode {episode}, reward:{self.reward}")
        plt.show()
        for key in self.traces:
            self.traces[key] = []
        self.rewards = []

    def get_parameters(self) -> dict[str, Any]:
        """Return the environment parameter settings as dict."""
        return {
            "min_pos": self.config.pos_range[0],
            "max_pos": self.config.pos_range[1],
            "min_speed": self.config.speed_range[0],
            "max_speed": self.config.speed_range[1],
            "reward-factors": self.reward_fac,
            "acceleration": self.config.acc,
            "step-size": self.dt,
            "reward_limit": self.reward_limit,
        }
