import logging
from collections import defaultdict
from collections.abc import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest
from gymnasium import spaces
from py_crane.boom import Wire
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv
from crane_controller.experiment_config import RewardConfig

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
        if (selection is not None and label in selection and selection[label] == 1) or selection is None:
            _ = ax1.plot(times, trace, label=label)
        elif label in selection and selection[label] == 2:
            _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    _ = plt.title(title)
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
        _ = crane.do_step(time, dt)
        yield (time + dt, crane)


def test_environment(
    crane: Callable[..., Crane],
    *,
    show: bool,
) -> None:
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=1.0,
            render_mode="plot" if show else "none",
            reward_limit=-0.01,
            discrete="energy",
            continuous_actions=False,
        ),
    )
    assert env.action_space.n == 3  # type: ignore[attr-defined,unused-ignore]
    assert env.action_space.start == 0  # type: ignore[attr-defined,unused-ignore]
    assert env.action_space.dtype == np.int64
    assert isinstance(env.action_space.seed(), int)
    assert len(env.observation_space.nvec) == 7, f"Found {env.observation_space.nvec}"  # type: ignore[attr-defined,union-attr]
    assert np.allclose(env.observation_space.nvec, [7, 4, 2, 2, 2, 2, 11])  # type: ignore[attr-defined,union-attr]
    assert np.allclose(env.observation_space.start, [0, 0, 0, 0, 0, 0, 0]), f"Found {env.observation_space.start}"  # type: ignore[attr-defined,union-attr]
    assert env.observation_space.dtype == np.int64
    assert isinstance(env.observation_space.seed(), int)
    q_values = defaultdict(lambda: np.array([0] * env.action_space.n))  # type: ignore[var-annotated,attr-defined]
    obs1 = np.array((0, 1, 1, 3, 0), int)
    obs2 = np.array((4, 0, 0, 1, 1), int)
    q_values[obs1.tobytes()]
    q_values[obs2.tobytes()]
    assert np.allclose(q_values[obs1.tobytes()], [0.0, 0.0, 0.0]), f"Found {q_values[obs1.tobytes()]}"
    assert q_values[obs2.tobytes()][2] == 0.0


def test_init(crane: Callable[..., Crane], *, show: bool) -> None:
    """Test the initialization of the environment."""
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            seed=1,
            start_speed=1.0,
            randomize_start=False,
            render_mode="play-back" if show else "data",
            continuous_actions=False,
        ),
    )
    assert isinstance(env.wire, Wire)
    rnd_u = env.np_random.uniform(2, 8)
    rnd_r = env.np_random.random()
    assert rnd_u == 5.07092974820154, f"Returns pseudo-random numbers when seed is given. Got {rnd_u} for seed 1"
    assert rnd_r == 0.9504636963259353, f"Returns pseudo-random numbers when seed is given. Got {rnd_r} for seed 1"
    obs, inf = env.reset(seed=1)
    # obs[3] is now pure theta_dot = cm_v[0] / wire.length (crane at rest so origin_v=0)
    assert len(obs) == 4
    assert np.isclose(obs[3], 1.0 / env.wire.length), f"Expected theta_dot=1/length, got obs[3]={obs[3]}"
    assert inf["steps"] == 0
    # reward = energy * 1.0 = -(0.5 * start_speed^2) at equilibrium (PE~0)
    assert abs(inf["reward"] + 0.5 * 1.0**2) < 1e-6, f"Found initial reward {inf['reward']}"
    obs, reward, terminated, truncated, _ = env.step(0)
    assert obs[0] == -0.1
    assert obs[1] == -0.1
    assert not terminated
    assert not truncated
    rewards: list[float] = []
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(int(env.np_random.integers(-1, 2)))
        rewards.append(reward)
    if show:
        show_figure(times=np.linspace(0, 100, 100), traces={"rewards": rewards})
    _ = env.reset()


def test_observation_space_dtype(crane: Callable[..., Crane]) -> None:
    """Test that the continuous observation space uses float64 dtype."""
    env = AntiPendulumEnv(crane, conf=None)
    assert env.observation_space.dtype == np.float64


def test_observations_are_float(crane: Callable[..., Crane]) -> None:
    """Test that observations preserve sub-integer precision after a physics step."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(continuous_actions=False))
    _ = env.reset()
    obs, _, _, _, _ = env.step(1)  # one physics step produces fractional values
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float64
    assert not np.all(obs == obs.astype(int))  # sub-integer precision is preserved


# ---------------------------------------------------------------------------
# rail_limit
# ---------------------------------------------------------------------------


def test_rail_limit_stored(crane: Callable[..., Crane]) -> None:
    """rail_limit is stored and bounds the continuous observation space."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(rail_limit=5.0))
    assert env.conf.rail_limit == 5.0
    assert env.spaces_min[0] == -5.0
    assert env.spaces_max[0] == 5.0


# ---------------------------------------------------------------------------
# obs[3] semantics: pure theta_dot
# ---------------------------------------------------------------------------


def test_obs3_is_pure_theta_dot(crane: Callable[..., Crane]) -> None:
    """obs[3] equals (cm_v[0] - origin_v[0]) / wire.length, not absolute velocity."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(start_speed=1.0, randomize_start=False))
    obs, _ = env.reset()
    wire = env.wire
    assert isinstance(wire, Wire)
    expected = (wire.cm_v[0] - wire.origin_v[0]) / wire.length  # pyright: ignore[reportUnknownMemberType]
    assert np.isclose(obs[3], expected)
    # At reset: crane at rest (origin_v=0), so theta_dot = start_speed / length
    assert np.isclose(obs[3], 1.0 / wire.length)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def test_reward_terms_zero_by_default(crane: Callable[..., Crane]) -> None:
    """New reward terms contribute zero when their weights are zero."""
    rc_energy = RewardConfig(energy=1.0, positional=0.0, time=0.0, position=0.0, acceleration=0.0)
    rc_crane_t = RewardConfig(energy=1.0, positional=0.0, time=100.0, position=0.0, acceleration=0.0)
    env1 = AntiPendulumEnv(
        crane, conf=AntiPendulumConfig(start_speed=1.0, reward_fac=rc_energy, continuous_actions=False)
    )
    env2 = AntiPendulumEnv(
        crane, conf=AntiPendulumConfig(start_speed=1.0, reward_fac=rc_crane_t, continuous_actions=False)
    )

    _ = env1.reset()
    _ = env2.reset()
    _, r1, _, _, _ = env1.step(2)
    _, r2, _, _, _ = env2.step(2)
    assert r1 > r2, f"crane_velocity=0 should give higher reward than crane_velocity=100; got r1={r1}, r2={r2}"


def test_crane_velocity_reward_term(crane: Callable[..., Crane]) -> None:
    """crane_velocity weight adds -crane_vel^2 to the reward."""
    rc = RewardConfig(energy=0.0, positional=0.0, position=0.0, acceleration=0.0, crane_velocity=-1.0)
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(start_speed=1.0, reward_fac=rc, continuous_actions=False))
    _ = env.reset()
    obs, reward, _, _, _ = env.step(2)  # max acceleration right
    crane_vel = obs[1]
    assert crane_vel != 0.0
    assert reward < 0.0, f"Found reward {reward}"
    assert np.isclose(reward, -(crane_vel**2), rtol=1e-4), f"Found {reward} != {-(crane_vel**2)}"


def test_terminal_penalty_on_truncation(crane: Callable[..., Crane]) -> None:
    """terminal_penalty is added to the reward when an episode truncates (OOB)."""
    rc = RewardConfig(energy=1.0, terminal_penalty=-50.0)
    env = AntiPendulumEnv(
        crane, conf=AntiPendulumConfig(start_speed=1.0, rail_limit=0.15, reward_fac=rc, continuous_actions=False)
    )
    _ = env.reset()
    got_truncation = False
    for _ in range(50):
        _, reward, _, truncated, _ = env.step(2)
        if truncated:
            assert reward < -40.0, f"Expected terminal penalty, got reward={reward}"
            got_truncation = True
            break
    assert got_truncation, "Expected at least one truncated step within 50 steps"


def test_action_space_is_discrete(crane: Callable[..., Crane]) -> None:
    """Part A: action space is Discrete(3) when continuous_actions=False."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(continuous_actions=False))
    assert isinstance(env.action_space, spaces.Discrete)
    assert int(env.action_space.n) == 3  # pyright: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Part B: continuous_actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuous_actions", [True, False])
def test_action_space_type(crane: Callable[..., Crane], continuous_actions: bool) -> None:  # noqa: FBT001
    """Action space is Box(-1,1) for continuous and Discrete(3) for discrete."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(continuous_actions=continuous_actions))
    if continuous_actions:
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == -1.0
        assert float(env.action_space.high[0]) == 1.0
    else:
        assert isinstance(env.action_space, spaces.Discrete)
        assert int(env.action_space.n) == 3  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("continuous_actions", [True, False])
def test_step_accepts_correct_action(crane: Callable[..., Crane], continuous_actions: bool) -> None:  # noqa: FBT001
    """step() accepts np.ndarray for continuous and int for discrete; obs shape unchanged."""
    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(continuous_actions=continuous_actions))
    _ = env.reset()
    if continuous_actions:
        action: int | np.ndarray = np.array([0.5], dtype=np.float32)
    else:
        action = 1
    obs, _, _, _, _ = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_environment(build_crane, show=True)
    # test_observation_space_dtype(build_crane)
    # test_reward_terms_zero_by_default(build_crane)
    # test_crane_velocity_reward_term(build_crane)
    # test_step_accepts_correct_action(build_crane, continuous_actions=True)
    # test_step_accepts_correct_action(build_crane, continuous_actions=False)
    # test_t_min_crane_reward_term(build_crane)
