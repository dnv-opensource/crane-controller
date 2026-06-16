"""Q-learning agent for the anti-pendulum environment."""

from __future__ import annotations

import datetime as dt
import json
import logging
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

    Args:
      values(Sequence[float] | np.ndarray): Raw data series to smooth.
      window(int): Number of elements in the averaging window.
      convolution_mode(valid", "same"}): Convolution mode passed to `numpy.convolve`.

    Returns
    -------
    Moving average as np array
    """
    return np.convolve(np.asarray(values, dtype=float).flatten(), np.ones(window), mode=convolution_mode) / window


class QLearningAgent:
    """Agent for training a controller via Q-learning.

    Args:
        env (AntiPendulumEnv): The environment instance to use
        learning_rate (float)=0.1: learning rate
        epsilon_decay (float)=1e-4: epsilon decay
        discount_factor (float)=0.95: Q-learning discound factor
        filename (Path): Optional file name (json file) to use as basis / save results
        use_file (str)='r': How to use the file (if provided). 'r', 'w' or 'rw'
        strategy (int)=0: Strategy to use:
           0: base strategy taken from gymnasium
           1: use current Q-values as histogram when choosing next action (not max and no learning rate) 
           2: alternative discrete space in environment: phase angle and moving average of acceleration
    """

    def __init__(
        self,
        env: AntiPendulumEnv,
        learning_rate: float = 0.1,
        epsilon_decay: float = 1e-4,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        filename: Path | None = None,
        use_file: str = "r",
        strategy: int = 0,
    ) -> None:
        """Initialize the Q-learning agent.

        See the class docstring for parameter descriptions.
        """
        self.env = env
        self.filename = Path(filename) if filename is not None else None
        self.use_file = use_file
        self.q_values: defaultdict[tuple[int, ...], np.ndarray]

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error: list[float] = []
        self.previous_steps = 0
        self.strategy = strategy
        #self.q_revised = defaultdict(lambda: np.array((0,) * self.env.action_space.n, int))

    def analyse_q(self, obs: tuple[int, ...]) -> None:
        """Log Q-table entries matching an observation pattern.

        Uses ``-1`` as a wildcard in the observation tuple to match any value
        in that dimension.

        Args:
          obs: tuple[int,...]: Observation tuple
        """
        for comb, q in self.q_values.items():
            include = not any(o >= 0 and o != c for c, o in zip(comb, obs, strict=True))
            if include:
                logger.info("%s %s %s %s %s", comb, q, int(np.argmax(q)), np.average(q), np.std(q) / np.average(q))

    def get_action(self, obs: tuple[int, ...]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Args:
          obs(tuple[int, ...]): Current discretised observation.

        Returns
        -------
        (int): action
        """
        if self.strategy != 1:
            if self.env.np_random.random() < self.epsilon:
                self.num_rnd += 1
                return self.env.action_space.sample()
            # With probability (1-epsilon): exploit (best known action)
            return np.argmax(self.q_values[obs])
        elif self.strategy == 1:
            _sum = 0.0
            for i,q in enumerate(self.q_values[obs]):
                if q == 0: # never calculated. We want all possibilities tried out
                    return i
                else:
                    _sum += q
            cum = []
            sum = 0.0
            for q in self.q_values[obs]:
                sum += q
                cum.append(sum)
            rnd = self.env.np_random.random()* _sum
            for i,c in enumerate(cum):
                if rnd <= c:
                    return i
            return len(cum)-1            

    def update_q(
        self,
        obs: tuple[int, ...],
        action: int,
        reward: float,
        *,
        terminated: bool,
        next_obs: tuple[int, ...],
        prev_reward: float
    ) -> bool:
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state).

        Args:
          obs (tuple[int,...]): the previoous observation
          action (int): the current action performed on obs
          reward (float): the reward from action, based on previous state (obs)
          terminated (bool): termination status after action
          next_obs (tuple[int,...]: Observation tuple after action
        """
        # What's the best we could do from the next state? Zero if episode terminated.
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value
        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]
        # Update our estimate in the direction of the error. Learning rate controls how big steps we take
        prev = self.q_values[obs].copy()
        lr = 1 if self.q_values[obs][action]==0.0 else self.lr # when no previous knowledge, avoid slow learning
        self.q_values[obs][action] = (1 - lr) * self.q_values[obs][action] + lr * temporal_difference
        
        #print(f"ACTION act:{action}: {prev} -> {self.q_values[obs]}")
        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)
        return np.argmax(self.q_values[obs])

    def episodes_init(self, n_episodes:int=1)-> tuple(int,int,int):
        """Perform initialization of episode."""
        if "r" in self.use_file and self.filename is not None and self.filename.exists():
            self.q_values = self.read_dumped(self.filename)
            logger.info("Starting %s episodes, using pre-trained values from %s", n_episodes, self.filename)
        else:  # start from scratch
            self.q_values = defaultdict(lambda: np.array((0.0,) * self.env.action_space.n, float))  # type: ignore[attr-defined,type-var]
            logger.info("Starting new training with %s episodes.", n_episodes)
        start_time = dt.datetime.now(dt.UTC)
        return (0,0,0)


    def do_episodes(self, n_episodes: int = 1000, max_steps: int = 5000, show: int = 0) -> None:
        """Run training or evaluation episodes.

        Uses pre-trained Q-values when available, otherwise starts a new
        training sequence.

        Args:
          n_episodes (int)=1000: Number of episodes to run
          max_steps (int)=5000: maximum number of steps before truncation
          show (int)=0: show mode (default no show)
        """
        start_time = dt.datetime.now(dt.timezone.utc)
        total_steps, num_terminated, num_truncated = self.episodes_init(n_episodes)
        rewards = [[], []]
        tau = []
        self.num_rnd = 0
        err_act = 0
        for _episode in tqdm(range(n_episodes)):
            # Start a new episode
            obs, _ = self.env.reset() # first reward is also available as self.env.reward
            assert isinstance(obs, tuple)
            num_failed, nsteps, terminated, truncated = (0, 0, False, False)

            #print(f"Episode {_episode}. Eps:{self.epsilon}, Q({obs}):{self.q_values[obs]}")
            while not terminated and not truncated and nsteps<max_steps:
                prev_reward = self.env.reward
                action = self.get_action(obs)  # choose action (initially random, gradually more intelligent)
                next_obs, _reward, terminated, truncated, _ = self.env.step(action)  # take action and observe result
                assert isinstance(next_obs, tuple)
                reward = float(_reward)
                _act = self.update_q(obs, action, reward, terminated=terminated, next_obs=next_obs, prev_reward=prev_reward)
                if _act != action: # q-tale revised such that max action changed
                    #self.q_revised[obs][_act] += 1
                    err_act += 1
                # Move to next state
                obs = next_obs
                nsteps += 1
            if show == SHOW_EPISODE_ANALYSIS:
                self.analyse_episode()
            num_terminated += int(terminated)
            num_truncated += int(truncated)
            if _episode >= n_episodes-100:
                log_r0 = np.log(-self.env.rewards[0])
                tau.append( np.average(np.array([-i*self.env.dt / (np.log(-r) -log_r0) for i,r in enumerate( self.env.rewards[1:])], float)))
                rewards[0].extend(list(range(len(self.env.rewards))))
                rewards[1].extend([np.log(-x)-log_r0 for x in self.env.rewards])
            total_steps += nsteps
            # Reduce exploration rate (agent becomes less random over time):
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        if show == SHOW_TRAINING_SUMMARY:
            self.analyse_training()
        if self.filename and "w" in self.use_file:
            self.dump_results(self.filename, n_episodes, total_steps, start_time, num_terminated, num_truncated)
        logger.info(f"Episodes:{n_episodes}, terminated:{num_terminated}, truncated:{num_truncated}")
        print(f"Steps:{total_steps}, revised actions:{err_act}, random actions:{self.num_rnd}")
        print(f"Term:{num_terminated}, trunc:{num_truncated}, tau:{np.average(tau)} +/-{np.std(tau)}")

        _,ax = plt.subplots(1,1)
        ax.plot( rewards[0], rewards[1], '.')
        plt.show()

    def dump_results(
        self,
        filename: str | Path = "",
        episodes: int = -1,
        steps: int = -1,
        start_time: dt.datetime | None = None,
        n_terminated: int = -1,
        n_truncated: int = -1,
    ) -> None:
        """Dump the Q-values to a JSON file.

        Args:
          filename(str|Path): Optional target file path.
             When empty, the filename provided at construction time is used (default "").
          episodes(int)=-1: the number of episodes which have been run
          steps(int)=-1: the limiting number of steps per episode
          start_time(dt.datetime)=None: clock-time when the training started
        """
        print("FILE", filename)
        if not filename:  # automatic file name
            if self.filename is None:
                logger.warning("No base file name provided. Aborting dump to file.")
                return
            _filename = self.filename
        else:
            _filename = Path(filename)

        converted: dict[str, list[float]] = {}
        for k, v in self.q_values.items():
            converted |= {str(k): list(v)}
        env_parameters = {k: str(v) for k, v in self.env.get_parameters().items()}
        content = {
            "start-training": "unknown" if start_time is None else start_time.strftime("%d.%m.%Y %H:%M:%S"),
            "end-training": dt.datetime.now(dt.UTC).strftime("%d.%m.%Y %H:%M:%S"),
            "pendulum": env_parameters,
            "q_agent": {
                "filename": str(self.filename),
                "use_file": self.use_file,
                "episodes": str(episodes),
                "steps": str(steps + self.previous_steps),
                "learning_rate": str(self.lr),
                "discount_factor": str(self.discount_factor),
                "epsilon-decay": str(self.epsilon_decay),
                "final-epsilon": str(self.final_epsilon),
                "epsilon": str(self.epsilon),
                "#terminated": n_terminated,
                "#truncated": n_truncated,
                "reward-trend": float(np.average(self.env.reward_stats[-100:][1])),
                "reward-std": float(np.average(self.env.reward_stats[-100:][2])),
                "reward-2nd": float(np.average(self.env.reward_stats[-100:][3])),                
            },
            "q_values": converted,
        }
        with _filename.open("w", encoding="utf-8") as _f:
            json.dump(content, _f, indent=3)
        logger.info("Updated q_values saved to %s", _filename.resolve())

    def read_dumped(self, filename: str | Path | None = None) -> defaultdict[tuple[int, ...], np.ndarray]:
        """Read a Q-values dict from a JSON file.

        Args:
          filename(str or Path): Path to the JSON file containing saved Q-values.

        Returns
        -------
        q_values dict
        """
        q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.array((0.0,) * self.env.action_space.n, float)  # type: ignore[attr-defined,type-var]
        )
        if filename is None and self.filename is None:  # there is no file to read. Return empty defautdict
            pass
        else:
            if filename is not None:
                path = Path(filename)
            elif self.filename is not None:
                path = Path(self.filename)

            with path.open(encoding="utf-8") as _f:
                from_dump = json.load(_f)
            self.previous_steps = int(from_dump["q_agent"]["steps"])
            self.epsilon = float(from_dump["q_agent"].get("epsilon", 1.0))
            self.epsilon_decay = float(from_dump["q_agent"].get("epsilon", 1e-4))
            assert "q_values" in from_dump, f"Key 'q_values' not found in file {filename}"
            for k, v in from_dump["q_values"].items():
                q_values.update({literal_eval(k): np.array(v) if isinstance(v, list) else v})
        return q_values

    def analyse_training(self, window: int = 10) -> None:
        """Plot moving averages of episode rewards, lengths, and training error.

        Args:
          window (int)=500: Moving average window size

        """
        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=3, figsize=(12, 5))

        lengths = [row[0] for row in self.env.reward_stats]
        rewards = [row[1] for row in self.env.reward_stats]

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = _get_moving_avgs(rewards, window // 10, "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = _get_moving_avgs(lengths, window // 10, "valid")
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

    def analyse_episode(self, window: int = 50) -> None:
        """Plot moving averages of rewards and training error for one episode.

        Args:
          window (int)=100: Moving average window size
        """
        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))

        rewards = _get_moving_avgs(self.env.rewards, window, "same")
        axs[0].set_title("Episode rewards")
        axs[0].plot(range(len(rewards)), rewards)
        axs[0].set_ylabel("rewards")
        axs[0].set_xlabel("Episode")

        axs[1].set_title("Training Error")
        training_error_mov_avg = _get_moving_avgs(self.training_error, window, "same")
        axs[1].plot(range(len(training_error_mov_avg)), training_error_mov_avg)
        axs[1].set_ylabel("Temporal Difference Error")
        axs[1].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def test_agent(self, num_episodes: int = 10) -> str:
        """Test agent performance without learning or exploration.

        Args:
          num_episodes (int) = 10: number of episodes to run.
        """
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
