from __future__ import annotations

import json
import logging
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv

logger = logging.getLogger(__name__)

SHOW_TRAINING_SUMMARY = 1
SHOW_EPISODE_ANALYSIS = 2


def _get_moving_avgs(
    values: Sequence[float] | np.ndarray,
    window: int,
    convolution_mode: Literal["valid", "same"],
) -> np.ndarray:
    """Compute moving averages to smooth noisy data.

    Parameters
    ----------
    values : Sequence[float] | np.ndarray
        Raw data series to smooth.
    window : int
        Number of elements in the averaging window.
    convolution_mode : {"valid", "same"}
        Convolution mode passed to `numpy.convolve`.

    Returns
    -------
    np.ndarray
        Smoothed data series.
    """
    return np.convolve(np.asarray(values, dtype=float).flatten(), np.ones(window), mode=convolution_mode) / window


class QLearningAgent:
    """Agent for training a controller via Q-learning.

    Parameters
    ----------
    env : AntiPendulumEnv
        The environment to be trained. Must provide `.reset()` and `.step()` methods.
    learning_rate : float, optional
        How quickly to update Q-values, in the range (0, 1] (default 0.1).
    initial_epsilon : float, optional
        Starting exploration rate (default 1.0).
    final_epsilon : float, optional
        Minimum exploration rate (default 0.1).
    discount_factor : float, optional
        How much to value future rewards, in the range [0, 1] (default 0.95).
    trained : tuple[str | Path, bool] or None, optional
        Optional path and flag for pre-trained Q-values.
        ``(filename, False)`` trains from scratch and saves;
        ``(filename, True)`` loads pre-trained values (default None).
    """

    DEFAULT_DISCRETE: ClassVar[dict[str, tuple[float | int, ...]]] = {
        "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
        "pos": (0, 1),
        "speed": (0, 1),
        "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
        "sector": (0, 1),
    }

    def __init__(
        self,
        env: AntiPendulumEnv,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        trained: tuple[str | Path, bool] | None = None,
    ) -> None:
        self.env = env
        _filename, self.use_pre_trained = trained if trained is not None else (None, False)
        self.filename: Path | None = Path(_filename) if _filename is not None else None
        self.q_values: defaultdict[tuple[int, ...], np.ndarray]
        if self.use_pre_trained and self.filename is not None:
            self.q_values = self.read_dumped(self.filename)
            self.epsilon = final_epsilon  # assume that we are fully learned
        else:  # start from scratch, but save the q_values afterwards
            self.q_values = defaultdict(lambda: np.array((0.0,) * env.action_space.n, float))  # type: ignore[attr-defined,type-var]
            self.epsilon = initial_epsilon  # start from scratch

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error: list[float] = []

    def analyse_q(self, obs: tuple[int, ...]) -> None:
        for comb, q in self.q_values.items():
            include = True
            for c, o in zip(comb, obs, strict=True):
                if o >= 0 and o != c:
                    include = False
                    break
            if include:
                logger.info("%s %s %s %s %s", comb, q, int(np.argmax(q)), np.average(q), np.std(q) / np.average(q))

    def get_action(self, obs: tuple[int, ...]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Parameters
        ----------
        obs : tuple[int, ...]
            Current discretised observation.

        Returns
        -------
        int
            Selected action index.
        """
        if self.env.np_random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        # With probability (1-epsilon): exploit (best known action)
        return int(np.argmax(self.q_values[obs]))

    def update_q(
        self,
        obs: tuple[int, ...],
        action: int,
        reward: float,
        *,
        terminated: bool,
        next_obs: tuple[int, ...],
    ) -> None:
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from
        (state, action, reward, next_state).

        See Also
        --------
        `Q-learning <https://en.wikipedia.org/wiki/Q-learning>`_

        Parameters
        ----------
        obs : tuple[int, ...]
            Previous observed state.
        action : int
            Action performed in state `obs`.
        reward : float
            Reward received after taking `action`.
        terminated : bool
            Whether the episode ended after `action`.
        next_obs : tuple[int, ...]
            New observed state after `action`.
        """
        # What's the best we could do from the next state? Zero if episode terminated.
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error. Learning rate controls how big steps we take
        self.q_values[obs][action] = (1 - self.lr) * self.q_values[obs][action] + self.lr * temporal_difference

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def do_episodes(self, n_episodes: int = 1000, max_steps: int = 5000, show: int = 0) -> None:
        """Run training or evaluation episodes.

        Uses pre-trained Q-values when available, otherwise starts a new
        training sequence.

        Parameters
        ----------
        n_episodes : int, optional
            Number of episodes to run (default 1000).
        max_steps : int, optional
            Maximum steps per episode before truncation (default 5000).
        show : int, optional
            Visualization mode - 0 for none, 1 for training summary, 2 for
            per-episode analysis (default 0).
        """
        if self.use_pre_trained:
            logger.info("Starting %s episodes, using pre-trained values from %s", n_episodes, self.filename)
        else:
            logger.info("Starting new training with %s episodes.", n_episodes)
        for _episode in tqdm(range(n_episodes)):
            # Start a new episode
            obs, _ = self.env.reset()
            assert isinstance(obs, tuple)
            nsteps = 0
            terminated, truncated = (False, False)

            while not terminated and not truncated:
                action = self.get_action(obs)  # choose action (initially random, gradually more intelligent)
                next_obs, _reward, terminated, truncated, _ = self.env.step(action)  # take action and observe result
                assert isinstance(next_obs, tuple)
                reward = float(_reward)
                self.update_q(obs, action, reward, terminated=terminated, next_obs=next_obs)
                # Move to next state
                obs = next_obs
                if show == SHOW_EPISODE_ANALYSIS:
                    self.analyse_episode()
                nsteps += 1
                truncated |= nsteps > max_steps
            # Reduce exploration rate (agent becomes less random over time):
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon / (n_episodes / 2))
        if show == SHOW_TRAINING_SUMMARY:
            self.analyse_training()
        if self.filename:
            self.dump_results()

    def dump_results(self, filename: str | Path = "") -> None:
        """Dump the Q-values to a JSON file.

        Parameters
        ----------
        filename : str or Path, optional
            Target file path. When empty, the filename provided at
            construction time is used (default "").
        """
        if not filename:  # automatic file name
            if self.filename is None:
                logger.warning("No base file name provided. Aborting dump to file.")
                return
            if self.use_pre_trained:  # do not overwrite pre-trained data
                if len(self.filename.stem.split("_")) == 1:
                    _filename = self.filename.parent / f"{self.filename.stem}_1{self.filename.suffix}"
                else:
                    stem, version = self.filename.stem.split("_")
                    _filename = self.filename.parent / f"{stem}_{int(version) + 1}{self.filename.suffix}"
            else:
                _filename = self.filename
        else:
            _filename = Path(filename)

        converted: dict[str, list[float]] = {}
        for k, v in self.q_values.items():
            converted.update({str(k): list(v)})
        with _filename.open("w", encoding="utf-8") as _f:
            json.dump(converted, _f, indent=3)
        logger.info("Updated q_values saved to %s", _filename.resolve())

    def read_dumped(self, filename: str | Path) -> defaultdict[tuple[int, ...], np.ndarray]:
        """Read a Q-values dict from a JSON file.

        Parameters
        ----------
        filename : str or Path
            Path to the JSON file containing saved Q-values.

        Returns
        -------
        defaultdict[tuple[int, ...], np.ndarray]
            Loaded Q-values mapping observation tuples to action-value arrays.
        """
        path = Path(filename)
        with path.open(encoding="utf-8") as _f:
            from_dump = json.load(_f)
        q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.array((0.0,) * self.env.action_space.n, float)  # type: ignore[attr-defined,type-var]
        )
        for k, v in from_dump.items():
            q_values.update({literal_eval(k): np.array(v) if isinstance(v, list) else v})
        return q_values

    def analyse_training(self, window: int = 500) -> None:

        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=3, figsize=(12, 5))

        lengths = [row[0] for row in self.env.reward_stats]
        rewards = [row[1] for row in self.env.reward_stats]

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = _get_moving_avgs(rewards, int(window / 10), "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = _get_moving_avgs(lengths, int(window / 10), "valid")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = _get_moving_avgs(self.training_error, window, "same")
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def analyse_episode(self, window: int = 100) -> None:

        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = _get_moving_avgs(self.env.rewards, window, "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[1].set_title("Training Error")
        training_error_moving_average = _get_moving_avgs(self.training_error, window, "same")
        axs[1].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[1].set_ylabel("Temporal Difference Error")
        axs[1].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def test_agent(self, num_episodes: int = 1000) -> str:
        """Test agent performance without learning or exploration."""
        total_rewards: list[float] = []

        # Temporarily disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            assert isinstance(obs, tuple)
            episode_reward = 0.0
            done = False

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                assert isinstance(next_obs, tuple)
                obs = next_obs
                episode_reward += float(reward)
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        self.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        msg = f"Test Results over {num_episodes} episodes:\n"
        msg += f"Win Rate: {win_rate:.1%}\n"
        msg += f"Average Reward: {average_reward:.3f}\n"
        msg += f"Standard Deviation: {np.std(total_rewards):.3f}\n"
        return msg
