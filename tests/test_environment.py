import logging
from collections import defaultdict
from collections.abc import Callable, Generator

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
    _, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None:
            _ = ax1.plot(times, trace, label=label)
        elif label in selection:
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
    _, _, _ = list(crane.booms())
    acc = -0.1
    crane.velocity = np.array((acc, 0.0, 0.0), float)
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
        if abs(time - int(time)) < 1e-6:
            acc = -acc
            crane.d_velocity[0] = acc
        crane.do_step(time, dt)
        yield (time + dt, crane)


def test_environment(
    crane: Callable[..., Crane],
    *,
    show: bool,
    v0: float,
    reward_limit: float,
) -> None:
    env = AntiPendulumEnv(
        crane,
        start_speed=v0,
        render_mode="plot" if show else "none",
        reward_limit=reward_limit,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    assert env.action_space.n == 3  # type: ignore[attr-defined,unused-ignore]
    assert env.action_space.start == 0  # type: ignore[attr-defined,unused-ignore]
    assert env.action_space.dtype == np.int64
    assert isinstance(env.action_space.seed(), int)
    assert len(env.observation_space.nvec) == 5  # type: ignore[attr-defined,unused-ignore]
    assert np.allclose(env.observation_space.nvec, [7, 2, 2, 6, 2])  # type: ignore[attr-defined,unused-ignore]
    assert np.allclose(env.observation_space.start, [0, 0, 0, 0, 0])  # type: ignore[attr-defined,unused-ignore]
    assert env.observation_space.dtype == np.int64
    assert isinstance(env.observation_space.seed(), int)
    q_values = defaultdict(lambda: np.array([env.low_reward()] * env.action_space.n))  # type: ignore[var-annotated]
    obs1 = np.array((0, 1, 1, 3, 0), int)
    obs2 = np.array((4, 0, 0, 1, 1), int)
    q_values[obs1.tobytes()]
    q_values[obs2.tobytes()]
    assert np.allclose(q_values[obs1.tobytes()], [-98.1000, -98.1000, -98.1000])
    assert q_values[obs2.tobytes()][2] == -98.1


def test_init(crane: Callable[..., Crane], *, show: bool) -> None:
    """Test the initialization of the environment."""
    env = AntiPendulumEnv(crane, seed=1, start_speed=1.0, render_mode="play-back" if show else "data")
    rnd_u = env.np_random.uniform(2, 8)
    rnd_r = env.np_random.random()
    assert rnd_u == 5.07092974820154, f"Returns pseudo-random numbers when seed is given. Got {rnd_u} for seed 1"
    assert rnd_r == 0.9504636963259353, f"Returns pseudo-random numbers when seed is given. Got {rnd_r} for seed 1"
    obs, inf = env.reset(seed=1)
    assert np.allclose(obs, [0.0, 0.0, np.pi, 0.017453292519943295]), f"Found {obs[3]}"
    assert inf["steps"] == 0
    assert abs(inf["reward"] + 0.5 * 0.017453292519943295**2) < 1e-9, f"Found initial reward {inf['reward']}"
    obs, reward, terminated, truncated, _ = env.step(-1)
    assert obs[0] == -0.1
    assert obs[1] == -0.1
    assert not terminated
    assert not truncated
    rewards = []
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(int(env.np_random.integers(-1, 2)))
        rewards.append(reward)
    if show:
        show_figure(times=np.linspace(0, 100, 100), traces={"rewards": rewards})
    env.reset()
