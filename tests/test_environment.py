import logging
from collections import defaultdict
from typing import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: F401
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


def show_figure(
    times: np.ndarray | list[float],
    traces: dict[str, list[float]] | dict[str, np.ndarray],
    selection: dict[str, int] | None = None,
    title: str = "",
):
    """Plot selected traces."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None:
            _ = ax1.plot(times, trace, label=label)
        else:
            if label in selection:
                if selection[label] == 1:
                    _ = ax1.plot(times, trace, label=label)
                elif selection[label] == 2:
                    _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    plt.title(title)
    plt.show()


def movement(crane: Crane, dt: float = 0.01, t_end: float = 10.0) -> Generator[tuple[float, Crane], None, None]:
    """Step through time yielding (time, crane) tuples with alternating acceleration."""
    f, p, w = list(crane.booms())
    acc = -0.1
    crane.velocity = np.array((acc, 0.0, 0.0), float)
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
        if abs(time - int(time)) < 1e-6:
            acc = -acc
            crane.d_velocity[0] = acc
        crane.do_step(time, dt)
        yield (time + dt, crane)


def test_environment(crane: Callable, show: bool, v0: float = 1.0, reward_limit=0.0):
    env = AntiPendulumEnv(
        crane,
        start_speed=v0,
        render_mode="plot" if show else "none",
        reward_limit=reward_limit,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    assert env.action_space.n == 3  # type: ignore[attr-defined]
    assert env.action_space.start == 0  # type: ignore[attr-defined]
    assert env.action_space.dtype == np.int64
    assert isinstance(env.action_space.seed(), int)
    assert len(env.observation_space.nvec) == 5  # type: ignore[attr-defined]
    assert np.allclose(env.observation_space.nvec, [7, 2, 2, 6, 2])  # type: ignore[attr-defined]
    assert np.allclose(env.observation_space.start, [0, 0, 0, 0, 0])  # type: ignore[attr-defined]
    assert env.observation_space.dtype == np.int64
    assert isinstance(env.observation_space.seed(), int)
    q_values = defaultdict(lambda: np.array([env.low_reward()] * env.action_space.n))  # type: ignore
    obs1 = np.array((0, 1, 1, 3, 0), int)
    obs2 = np.array((4, 0, 0, 1, 1), int)
    q_values[obs1.tobytes()]
    q_values[obs2.tobytes()]
    assert np.allclose(q_values[obs1.tobytes()], [-98.1000, -98.1000, -98.1000])
    assert q_values[obs2.tobytes()][2] == -98.1


def test_observation_space_dtype(crane: Callable):
    env = AntiPendulumEnv(crane)
    assert env.observation_space.dtype == np.float64


def test_observations_are_float(crane: Callable):
    env = AntiPendulumEnv(crane)
    env.reset()
    obs, _, _, _, _ = env.step(1)  # one physics step produces fractional values
    assert obs.dtype == np.float64
    assert not np.all(obs == obs.astype(int))  # sub-integer precision is preserved


def test_init(crane: Crane, show: bool = False):
    """Test the initialization of the environment."""
    env = AntiPendulumEnv(crane, seed=1, start_speed=-1.0, render_mode="play-back" if show else "data")
    rnd_u = env.np_random.uniform(2, 8)
    rnd_r = env.np_random.random()
    assert rnd_u == 5.07092974820154, f"Returns pseudo-random numbers when seed is given. Got {rnd_u} for seed 1"
    assert rnd_r == 0.9504636963259353, f"Returns pseudo-random numbers when seed is given. Got {rnd_r} for seed 1"
    obs, inf = env.reset()
    assert np.allclose(obs, [0.0, 0.0, 0.0, -0.7405126971046593]), f"Found {obs}"
    assert inf["steps"] == 0
    assert abs(inf["reward"] + 0.5 * (-0.7405126971046593) ** 2) < 1e-9, f"Found initial reward {inf['reward']}"
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs[0] == -1e-5, f"Found {obs[0]}"
    assert obs[1] == -0.001, f"Found {obs[1]}"
    assert not terminated
    assert not truncated
    rewards = []
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.np_random.integers(0, 3))
        rewards.append(reward)
    if show:
        show_figure(times=np.linspace(0, 100, 100), traces={"rewards": rewards})
    env.reset()


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest  # noqa: F401

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_init(build_crane, show=True)
    # test_environment(build_crane, show=True)
