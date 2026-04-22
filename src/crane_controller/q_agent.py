import json
import logging
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm  # Progress bar

logger = logging.getLogger(__name__)


class QLearningAgent(object):
    """Agent for training the controller (a Gym Environment).

    Args:
        env (gym.Env): The Environment (class) to be trained. Need .reset() and .step() functions.
        learning_rate (float): How quickly to update Q-values (0-1)
        initial_epsilon (float): Starting exploration rate (usually 1.0)
        final_epsilon (float): Minimum exploration rate (usually 0.1)
        discount_factor (float): How much to value future rewards (0-1)
        trained (tuple[str,bool]): Optional possibility to save q_values after training / read pre-trained q_values:
           (filename,use-it): (filename,False): perform new training and save, (filename,True) use pre-trained values
    """

    DEFAULT_DISCRETE: dict = {
        "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
        "pos": (0, 1),
        "speed": (0, 1),
        "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
        "sector": (0, 1),
    }

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        trained: tuple[str | Path, bool] | None = None,
    ):
        self.env = env
        # print("ACTION_SPACE.N", env.action_space.n, defaultdict( lambda: np.zeros(env.action_space.n))['xx'])
        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        _filename, self.use_pre_trained = trained if trained is not None else (None, False)
        self.filename: Path | None = Path(_filename) if _filename is not None else None
        if self.use_pre_trained and self.filename is not None:
            self.q_values = self.read_dumped(self.filename)
            self.epsilon = final_epsilon  # assume that we are fully learned
        else:  # start from scratch, but save the q_values afterwards
            self.q_values = defaultdict(lambda: np.array((0.0,) * env.action_space.n, float))  # type: ignore  ## n!
            self.epsilon = initial_epsilon  # start from scratch

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error: list[float] = []

    def analyse_q(self, obs: tuple[int, ...]):
        for comb, q in self.q_values.items():
            include = True
            for c, o in zip(comb, obs, strict=True):
                if o >= 0 and o != c:
                    include = False
                    break
            if include:
                print(comb, q, int(np.argmax(q)), np.average(q), np.std(q) / np.average(q))

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns
        -------
            action: 0 (stand) or 1 (hit)
        """
        if np.random.random() < self.epsilon:  # With probability epsilon: explore (random action)
            return self.env.action_space.sample()
        else:  # With probability (1-epsilon): exploit (best known action)
            return int(np.argmax(self.q_values[obs]))

    def update_q(
        self,
        obs: gym.Space,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: gym.Space,
    ):
        """Update Q-value based on experience.
        self.update(obs, action, reward, terminated, next_obs) # learn from this experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state).

        See also `Q-learning <https://en.wikipedia.org/wiki/Q-learning>`_

        Args:
            obs: previous observed state
            action: action performed on the state 'obs'
            reward: the reward from 'action'
            next_obs: the new observed state after 'action' on state 'obs'
            terminated: info whether the agent was terminated after 'action'
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

    def do_episodes(self, n_episodes: int = 1000, max_steps: int = 5000, show: int = 0):
        """Do n_episodes, using pre-trained q_values or starting a new training sequence."""
        if self.use_pre_trained:
            logger.info(f"Starting {n_episodes} episodes, using pre-trained values from {self.filename}")
        else:
            logger.info(f"Starting new training with {n_episodes} episodes.")
        for _episode in tqdm(range(n_episodes)):
            # Start a new episode
            obs, info = self.env.reset()
            nsteps = 0
            terminated, truncated = (False, False)

            while not terminated and not truncated:
                action = self.get_action(obs)  # choose action (initially random, gradually more intelligent)
                next_obs, _reward, terminated, truncated, info = self.env.step(action)  # take action and observe result
                reward = float(_reward)
                self.update_q(obs, action, reward, terminated, next_obs)  # learn from this experience
                # Move to next state
                obs = next_obs
                if show == 2:
                    self.analyse_episode()
                nsteps += 1
                truncated |= nsteps > max_steps
            # Reduce exploration rate (agent becomes less random over time):
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon / (n_episodes / 2))
        if show == 1:
            self.analyse_training()
        if self.filename:
            self.dump_results()

    def dump_results(self, filename: str | Path = ""):
        """Dump the q_values to a json file."""
        if not filename:  # automatic file name
            if self.filename is None:
                logger.warning("No base file name provided. Aborting dump to file.")
                return
            elif self.use_pre_trained:  # do not overwrite pre-trained data
                if len(self.filename.stem.split("_")) == 1:
                    _filename = self.filename.parent / f"{self.filename.stem}_1{self.filename.suffix}"
                else:
                    stem, version = self.filename.stem.split("_")
                    _filename = self.filename.parent / f"{stem}_{int(version) + 1}{self.filename.suffix}"
            else:
                _filename = self.filename
        else:
            _filename = Path(filename)

        converted: dict[str, list] = {}
        for k, v in self.q_values.items():
            converted.update({str(k): list(v) if isinstance(v, np.ndarray) else v})
        with open(_filename, "w") as _f:
            json.dump(converted, _f, indent=3)
        logger.info(f"Updated q_values saved to {_filename.resolve()}")

    def read_dumped(self, filename: str | Path):
        """Read a q_values dict (saved as json) from file."""
        with open(filename, "r") as _f:
            from_dump = json.load(_f)
        q_values = defaultdict(lambda: np.array((0.0,) * self.env.action_space.n, float))  # type: ignore # n exists
        for k, v in from_dump.items():
            q_values.update({literal_eval(k): np.array(v) if isinstance(v, list) else v})
        return q_values

    def analyse_training(self, window: int = 500):

        def get_moving_avgs(arr, window, convolution_mode):
            """Compute moving average to smooth noisy data."""
            return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window

        # Smooth over the given episode window
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

        lengths = [row[0] for row in self.env.reward_stats]  # type: ignore  ## reward_stats exist
        rewards = [row[1] for row in self.env.reward_stats]  # type: ignore  ## reward_stats exist

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = get_moving_avgs(rewards, int(window / 10), "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = get_moving_avgs(lengths, int(window / 10), "valid")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = get_moving_avgs(self.training_error, window, "same")
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def analyse_episode(self, window: int = 100):

        def get_moving_avgs(arr, window, convolution_mode):
            """Compute moving average to smooth noisy data."""
            return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window

        # Smooth over the given episode window
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = get_moving_avgs(self.env.rewards, window, "valid")  # type: ignore  ## rewards exist
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[1].set_title("Training Error")
        training_error_moving_average = get_moving_avgs(self.training_error, window, "same")
        axs[1].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[1].set_ylabel("Temporal Difference Error")
        axs[1].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def test_agent(self, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
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
