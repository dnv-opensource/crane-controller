"""Q-learning agent for the anti-pendulum environment."""

from __future__ import annotations

import datetime as dt
import json
import logging
from ast import literal_eval
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv
from crane_controller.experiment_config import RewardConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gymnasium as gym

LOGGER = logging.getLogger(__name__)

SHOW_TRAINING_SUMMARY = 1
SHOW_EPISODE_ANALYSIS = 2


@dataclass(kw_only=True, frozen=True, slots=True)
class QLearningConfig:
    """Hyperparameters for Q-learning.

    Args:
        learning_rate (float) = 0.1: learning rate (how much q-update vs. use old),
        epsilon_decay (float)=1e-4: transition from initial to final epsilon
        final_epsilon: float = 0.1,
        discount_factor (float)=0.95: Q-learning discount factor
        q_default: default value used when row in Q-table is created.
        strategy: 'default', 'q-hist', 'r-trend'
        filename: name of file to read from and/or save to
        use_file: how to use the file (if not None): 'r', 'w', 'rw'
        episodes: number of episodes already run (default:0)
        steps: number of steps already run (default:0)
        num_terminated: number of terminated episodes within run episodes
        num_truncated: number of truncated episodes within run episodes
        auto_run: optionally provide the number of episodes to run on this configuration (default none)
        max_steps: optionally provide the maximum number of steps to use in auto_run
        start_training: optional start time of training (when reading trained data sets)
        end_training: optional end time of training (when reading trained data sets)
    """

    learning_rate: float = 0.1
    epsilon_decay: float = 1e-4
    final_epsilon: float = 0.1
    discount_factor: float = 0.95
    q_default: float | None = None
    strategy: str = "default"
    # additional information, especially for continued training or use of trained values
    filename: str | None = None
    use_file: str = "r"
    episodes: int = 0
    steps: int = 0
    epsilon: float = 1.0
    num_terminated: int = 0
    num_truncated: int = 0
    auto_run: int = 0
    max_steps: int = 1000
    start_training: str = "unknown"
    end_training: str = "unknown"


class QLearningAgent:
    """Agent for training a controller via Q-learning."""

    STAT_LEN = 500

    def __init__(
        self,
        env: gym.Env[tuple[int, ...] | np.ndarray, int],
        conf: QLearningConfig | None = None,
        q_values: defaultdict[tuple[int, ...], np.ndarray] | None = None,
        filename: Path | None = None,
        use_file: str | None = None,
    ) -> None:
        """Initialize the Q-learning agent.

        Args:
            env: The environment instance to use
            conf: configuration of Q-learning, or use default values
            q_values: Optionally provide q_values (e.g. read from file)
            filename: Optional possibility to configure the agent from a configuration json file
            use_file: information to the agent how to use a file. 'r', 'w' or 'rw'. Default 'r'
        """
        self.env = env
        self.conf = QLearningConfig() if conf is None else conf
        self.filename = (
            filename if filename is not None else (Path(self.conf.filename) if self.conf.filename is not None else None)
        )
        self.use_file = use_file if use_file is not None else self.conf.use_file  # base value might be changed

        self.q_values: defaultdict[tuple[int, ...], np.ndarray]
        if q_values is not None:  # explicitly supplied, e.g. from saved training
            self.q_values = q_values
        else:  # start from scratch
            self.q_values = defaultdict(lambda: np.array((self.conf.q_default,) * self.env.action_space.n, float))  # type: ignore[attr-defined,type-var]

        self.epsilon = self.conf.epsilon  # default value or from pre-trained data
        self.epsilon_decay = self.conf.epsilon_decay  # default value. May be changed when reading from file

        # Track learning progress
        self.num_rnd = 0
        self.training_error: list[float] = []
        if self.conf.auto_run == 1:  # one deterministic episode
            self.deterministic_episode(max_steps=self.conf.max_steps)
        elif self.conf.auto_run > 1:
            self.do_episodes(n_episodes=self.conf.auto_run, max_steps=self.conf.max_steps)

    def analyse_q(self, obs: tuple[int, ...]) -> None:
        """Log Q-table entries matching an observation pattern.

        Uses ``-1`` as a wildcard in the observation tuple to match any value
        in that dimension.

        Args:
          obs: Observation tuple
        """
        for comb, q in self.q_values.items():
            include = not any(o >= 0 and o != c for c, o in zip(comb, obs, strict=True))
            if include:
                LOGGER.info("%s %s %s %s %s", comb, q, int(np.argmax(q)), np.average(q), np.std(q) / np.average(q))

    def _check_default(self, obs: tuple[int, ...]) -> int:
        for i, q in enumerate(self.q_values[obs]):
            if np.isnan(q) or q == self.conf.q_default:
                return i
        return -1

    def get_action(self, obs: tuple[int, ...]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Args:
          obs(tuple[int, ...]): Current discretised observation.

        Returns:
        -------
        (int): action
        """
        if self.conf.strategy == "q-hist":  # random action with weight on heighest Q-value
            q_sum = 0.0
            for i, q in enumerate(self.q_values[obs]):  # type: ignore[index]
                if np.isnan(q) or q == self.conf.q_default:  # never calculated. We want all possibilities tried out
                    return i
                q_sum += q
            cum = []
            _sum = 0.0
            for q in self.q_values[obs]:  # type: ignore[index]
                _sum += q
                cum.append(_sum)
            rnd = self.env.np_random.random() * q_sum
            for i, c in enumerate(cum):
                if rnd <= c:
                    return i
            return len(cum) - 1

        if self.env.np_random.random() < self.epsilon:
            self.num_rnd += 1
            return self.env.action_space.sample()
        # With probability (1-epsilon): exploit (best known action)
        i_default = self._check_default(obs)
        return i_default if i_default >= 0 else np.argmax(self.q_values[obs])  # type: ignore[return-value,index]

    def update_q(
        self,
        s0: tuple[int, ...],
        a0: int,
        r1: float,
        *,
        terminated: bool,
        s1: tuple[int, ...],
        r0: float,
    ) -> bool:
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, a0, r1, next_state).

        Args:
          s0: the previoous state (observation)
          a0: the current action performed on s0
          r1: the reward from action a0, based on previous state (s0)
          terminated: termination status after action a0
          s1: Observation tuple after a0 from s0
          r0: Previous reward, leading to s0
        """
        lr = self.conf.learning_rate  # the learning rate
        q_s0_a0 = self.q_values[s0][a0]  # current Q-value
        q_s1_max = (not terminated) * np.max(self.q_values[s1])  # type: ignore[index]
        if np.isnan(q_s1_max):
            q_s1_max = 0.0
        if self.conf.strategy == "r-trend":
            q_target = (1 - self.conf.discount_factor) * r1 + self.conf.discount_factor * q_s1_max
        elif q_s1_max == self.conf.q_default:
            q_target = (r1 - r0) + self.conf.discount_factor * r1
        else:
            q_target = (r1 - r0) + self.conf.discount_factor * q_s1_max  # estimate of optimal Q
        if q_s0_a0 == self.conf.q_default or np.isnan(q_s0_a0):  # no previous knowledge. Do 'fast forward' learning
            self.q_values[s0][a0] = q_target
        else:  # otherwise we update according to the learning rate
            self.q_values[s0][a0] = (1 - lr) * q_s0_a0 + lr * q_target

        self.training_error.append(q_target - q_s0_a0)  # Track learning progress (useful for debugging)
        return np.argmax(self.q_values[s0])  # type: ignore[return-value,index]

    def do_episodes(self, n_episodes: int = 1000, max_steps: int = 5000, show: int = 0) -> None:
        """Run training or evaluation episodes.

        Uses pre-trained Q-values when available, otherwise starts a new
        training sequence.

        Args:
          n_episodes: Number of episodes to run
          max_steps: maximum number of steps before truncation
          show: show mode (default no show)
        """
        start_time = dt.datetime.now(dt.UTC)
        total_steps = 0
        num_terminated = self.conf.num_terminated
        num_truncated = self.conf.num_truncated
        rewards: list[list[float]] = [[], []]
        tau: list[float] = []
        self.num_rnd = 0
        err_act = 0
        obs, _ = self.env.reset(options={"init": True})  # initial reset. No show. calc first obs and reward
        assert isinstance(obs, tuple)
        for _episode in tqdm(range(n_episodes)):
            # Start a new episode
            assert isinstance(obs, tuple), f"Found {obs}. Expected tuple of float"

            LOGGER.debug(f"Episode {_episode}. Eps:{self.epsilon}, Q({obs}):{self.q_values[obs]}")
            for _nsteps in range(max_steps):
                prev_reward = self.env.reward  # type: ignore[attr-defined] ## extended class
                action = self.get_action(obs)  # choose action (initially random, gradually more intelligent)
                next_obs, _reward, term, trunc, _ = self.env.step(action)  # take action and observe result
                assert isinstance(next_obs, tuple)
                reward = float(_reward)
                _act = self.update_q(obs, action, reward, terminated=term, s1=next_obs, r0=prev_reward)
                if _act != action:  # q-tale revised such that max action changed
                    err_act += 1
                if term or trunc:
                    break
                # Move to next state
                obs = next_obs
            if show == SHOW_EPISODE_ANALYSIS:
                self.analyse_episode()
            num_terminated += int(term)
            num_truncated += int(trunc)
            if _episode >= n_episodes - 100:
                log_r0 = np.log(np.abs(self.env.rewards[0]))  # type: ignore[attr-defined] ## extended class
                _env_dt = getattr(getattr(self.env, "conf", self.env), "dt", 1.0)
                _t = [-i * _env_dt / (np.log(np.abs(r)) - log_r0) for i, r in enumerate(self.env.rewards[1:])]  # type: ignore[attr-defined] ## extended class
                tau.append(np.average(_t))
                rewards[0].extend(list(range(len(self.env.rewards))))  # type: ignore[attr-defined] ## extended class
                rewards[1].extend([np.log(np.abs(x)) - log_r0 for x in self.env.rewards])  # type: ignore[attr-defined] ## extended class
            total_steps += _nsteps
            # Reduce exploration rate (agent becomes less random over time):
            self.epsilon = max(self.conf.final_epsilon, self.epsilon - self.epsilon_decay)
            obs, _ = self.env.reset()  # reset after episode, show, calc obs and reward for next episode.
            assert isinstance(obs, tuple)

        if self.filename and "w" in self.use_file:
            self.dump_results(self.filename, n_episodes, total_steps, start_time, num_terminated, num_truncated)
        LOGGER.info(f"Episodes:{n_episodes}, terminated:{num_terminated}, truncated:{num_truncated}")
        LOGGER.info(f"Steps:{total_steps}, revised actions:{err_act}, random actions:{self.num_rnd}")
        LOGGER.info(f"Term:{num_terminated}, trunc:{num_truncated}, tau:{np.average(tau)} +/-{np.std(tau)}")

        if show == SHOW_TRAINING_SUMMARY:
            self.analyse_training()

            _, ax = plt.subplots(1, 1)
            ax.plot(rewards[0], rewards[1], ".")
            plt.show()

    def deterministic_episode(self, max_steps: int = 1000) -> None:
        """Run one episode using pre-trained q_table values. Random only if row does not yet exist.

        Args:
          max_steps: maximum number of steps before truncation
          show: show mode (default no show)
        """
        obs, _ = self.env.reset(seed=self.env.conf.seed, options={"init": True})  # type: ignore[attr-defined]
        _r0 = self.env.reward  # type: ignore[attr-defined]

        # Start a new episode
        assert isinstance(obs, tuple)

        for _ in range(max_steps):
            try:
                assert isinstance(obs, tuple)
                action = np.nanargmax(self.q_values[obs])
            except ValueError:
                break  # all-NaN slice
            _obs, _reward, term, trunc, _ = self.env.step(int(action))  # take action and observe result
            obs = _obs
            if term or trunc:
                break

        obs, _ = self.env.reset()  # show. calc first obs and reward

    def dump_results(
        self,
        filename: str | Path | None = None,
        episodes: int = 0,
        steps: int = 0,
        start_time: dt.datetime | None = None,
        n_terminated: int = 0,
        n_truncated: int = 0,
    ) -> None:
        """Dump the Q-values to a JSON file.

        Args:
          filename: Optional target file path.
             When empty, the filename provided at construction time is used (default "").
          episodes: the number of episodes which have been run
          steps: the limiting number of steps per episode
          start_time: clock-time when the training started
          n_terminated: number of terminated episodes
          n_truncated: number of truncated episodes
        """
        if not filename:  # automatic file name
            if self.filename is None:
                if self.conf.filename is not None:
                    _filename = Path(self.conf.filename)
                else:
                    LOGGER.warning("No base file name provided. Aborting dump to file.")
                    return
            else:
                _filename = Path(self.filename)
        else:
            _filename = Path(filename)

        converted: dict[str, list[float]] = {}
        for k, v in self.q_values.items():
            converted |= {str(k): list(v)}
        # keep start- and end-time if nothing is done
        t_start = (
            self.conf.start_training
            if episodes == 0
            else ("unknown" if start_time is None else start_time.strftime("%d.%m.%Y %H:%M:%S"))
        )
        t_end = self.conf.end_training if episodes == 0 else dt.datetime.now(dt.UTC).strftime("%d.%m.%Y %H:%M:%S")
        q_agent = asdict(self.conf)
        # some special considerations for some fields (values may change during more training):
        q_agent["start_training"] = t_start
        q_agent["end_training"] = t_end
        q_agent["filename"] = str(_filename)
        q_agent["use_file"] = self.use_file
        q_agent["episodes"] = episodes + self.conf.episodes
        q_agent["steps"] = steps + self.conf.steps
        q_agent["epsilon_decay"] = self.epsilon_decay
        q_agent["epsilon"] = self.epsilon
        q_agent["num_terminated"] = self.conf.num_terminated + n_terminated
        q_agent["num_truncated"] = self.conf.num_truncated + n_truncated

        content = {
            "environment": self.env.get_parameters(),  # type: ignore[attr-defined]
            "q_agent": q_agent,
            "q_values": converted,
        }
        with _filename.open("w", encoding="utf-8") as _f:
            json.dump(content, _f, indent=3)
        LOGGER.info("Updated q_values saved to %s", _filename.resolve())

    @staticmethod
    def read_dumped(
        filename: str | Path,
    ) -> tuple[dict[str, Any], defaultdict[tuple[int, ...], np.ndarray]]:
        """Read parameters and the Q-values dict from a JSON file.

        If the file is read before the instantiation of environment and agent,
        both can be configured with the dicts return from the file.

        Args:
          filename(str or Path): Path to the JSON file containing saved Q-values.

        Returns:
        -------
        properties and q_values dict
        """
        if not filename:  # there is no file to read. Return empty defautdict
            raise ValueError("No valid filename provided.")
        path = Path(filename)
        if not path.exists():
            raise ValueError("Required configuration/q_values file {filename} not found.")

        with path.open(encoding="utf-8") as _f:
            info = json.load(_f)
        assert "q_values" in info, f"Key 'q_values' not found in file {filename}"
        _q = info.pop("q_values")
        q_len: int = 0
        for v in _q.values():
            if q_len == 0:
                q_len = len(v)
            else:
                assert len(v) == q_len, f"Q-values are not of equal length. {q_len} expected."
        q_default = info["q_agent"]["q_default"] if info["q_agent"]["q_default"] is not None else np.nan
        q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.array((q_default,) * q_len))
        for k, v in _q.items():
            q_values.update({literal_eval(k): np.array(v) if isinstance(v, list) else v})

        if info["environment"]["reward_fac"] is None:
            info["environment"]["reward_fac"] = asdict(RewardConfig())  # default reward factors
        return (info, q_values)

    @staticmethod
    def auto_run(  # noqa: PLR0912, C901
        conf: Path | dict[str, Any], _env: dict[str, Any] | None = None, _agent: dict[str, Any] | None = None
    ) -> QLearningAgent:
        """Perform training/evaluation on the (Anti-)Pendulum environment using q-learning.

        A file is expected which at least specifies the environment and q_agent configuration.
        q_values might be empty if a new learning session is to be started

        Args:
            conf: basic configuration provided as file Path (json file) or dict
            _env: optional adaptation of environment configuration parameters as dict
            _agent: optional adaptation of q_agent configuration parameters as dict
        """
        if isinstance(conf, dict):
            for k in ("environment", "q_agent", "q_values"):
                assert k in conf, "Mandory key {k} not found in {conf}"
            _q_values = conf.pop("q_values")
            assert isinstance(_q_values, dict)
            q_len = 0
            for v in _q_values.values():
                if q_len == 0:
                    q_len = len(v)
                else:
                    assert len(v) == q_len, "Varying vector length in q_values"
            q_default = conf.get("q_default", None)
            if q_default is None:
                q_default = np.nan
            q_values: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.array((q_default,) * q_len))
            for k, v in _q_values:
                q_values.update({literal_eval(k): np.array(v) if isinstance(v, list) else v})
            assert all(isinstance(k, tuple) and isinstance(v, np.ndarray) for k, v in q_values.items())
        elif isinstance(conf, Path):
            assert conf.exists(), f"File {conf} not found"
            filename = conf
            conf, q_values = QLearningAgent.read_dumped(conf)
            conf["q_agent"]["filename"] = Path(filename)
        else:
            raise TypeError(f"conf must either be a dict or a Path object. Found {conf}") from None
        if _env is not None:
            for k, v in _env.items():
                conf["environment"][k] = v
        if _agent is not None:
            for k, v in _agent.items():
                conf["q_agent"][k] = v
        env = AntiPendulumEnv(build_crane, conf=AntiPendulumConfig(**conf["environment"]))
        agent = QLearningAgent(
            env,
            conf=QLearningConfig(**conf["q_agent"]),
            q_values=q_values,
        )
        if agent.conf.filename is not None and "w" in agent.use_file and conf["q_agent"]["auto_run"] != 0:
            LOGGER.info(f"Model saved to {agent.conf.filename}")
        return agent

    def analyse_training(self, window: int = 10) -> None:
        """Plot moving averages of episode rewards, lengths, and training error.

        Args:
          window: Moving average window size
        """
        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=3, figsize=(12, 5))

        lengths = self.env.reward_stats["steps"]  # type: ignore[attr-defined] ## extended class
        rewards = self.env.reward_stats["reward"]  # type: ignore[attr-defined] ## extended class

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
          window: Moving average window size
        """
        # Smooth over the given episode window
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))

        rewards = _get_moving_avgs(self.env.rewards, window, "same")  # type: ignore[attr-defined] ## extended class
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
            num_episodes: number of episodes to run.

        Returns:
            (str) result message.
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

    Returns:
    -------
    Moving average as np array
    """
    return np.convolve(np.asarray(values, dtype=float).flatten(), np.ones(window), mode=convolution_mode) / window
