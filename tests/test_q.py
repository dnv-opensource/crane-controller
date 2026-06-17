import itertools
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from py_crane.boom import Wire
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv, _level
from crane_controller.experiment_config import RewardConfig
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


def test_smoke(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="plot" if show else "none",
            reward_limit=-0.05,
            discrete="energy",
            continuous_actions=False,
        ),
    )
    agent = QLearningAgent(env, filename=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


@pytest.mark.skip(reason="Test must be updated")
def test_q_analyse(crane: Callable[..., Crane], *, show: bool) -> None:
    models = Path(__file__).parent.resolve().parent / "models"
    assert (models / "q_trained.json").exists(), "Expect a file 'q_trained.json' in the models directory. Not found"
    _ = shutil.copy2(models / "q_trained.json", ".")  # copy to working_directory
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            discrete="energy",
            continuous_actions=False,
        ),
    )
    assert Path("q_trained.json").exists(), "File 'q_trained.json' not found"
    agent = QLearningAgent(env, filename=Path("q_trained.json"), use_file="r")
    agent.q_values = agent.read_dumped()
    for k, v in agent.q_values.items():
        assert len(k) == 5, len(v) == 3
    for pos in (0, 1):
        for speed in (0, 1):
            res = {k: v for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed}
            logger.info(f"pos:{pos}, speed:{speed}")
            acc: list[np.floating] = []
            for i in range(3):
                col = [x[i] for x in res.values()]
                acc.append(np.average(col))
            logger.info(f"averages: {acc}")


@pytest.mark.skip(reason="Test must be updated")
def test_intervals(crane: Callable[..., Crane]):
    """Test that learning / saving / resuming learning works:"""
    save_path = Path.cwd() / "q_interval_training.json"
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="none",
            reward_limit=-0.05,
            discrete="energy",
            continuous_actions=False,
        ),
    )

    agent = QLearningAgent(env, filename=save_path, use_file="w")
    for i in range(10):
        _ = env.reset(seed=i + 1)
        agent.do_episodes(n_episodes=2, max_steps=100)
        if i == 0:
            agent = QLearningAgent(env, filename=save_path, use_file="rw")
    logger.info(f"Model saved to {save_path}")


@pytest.mark.skip(reason="Test must be updated")
def test_levels(crane: Callable[..., Crane]) -> None:
    def check(val: float, expected: int) -> None:
        assert _level(val, env.discrete["energy"]) == expected, f"Level {val} =? {_level(val, env.discrete['energy'])}"

    env = AntiPendulumEnv(crane)
    logger.info(env.discrete)
    check(0, 0)
    check(-1e-10, -1)
    check(1e-10, 0)
    check(0.014, 0)
    check(0.015, 1)
    check(0.3, 1)
    check(0.4, 2)
    check(1.4, 2)
    check(1.5, 3)
    check(5.9, 3)
    check(6.0, 4)
    check(13.1, 4)
    check(13.2, 5)
    check(98, 5)
    check(99, 6)
    check(float("inf"), 6)


@pytest.mark.skip(reason="Test must be updated")
def test_discretization(crane: Callable[..., Crane], *, show: bool, discretization: str) -> None:
    """Test the discretization with respect to yielding unique rewards."""
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=2.0,
            render_mode="none",
            reward_limit=0.0,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.01}),
            discrete=discretization,
        ),
    )
    env.reset()
    _agent = QLearningAgent(env)
    for e in range(len(env.discrete["energy"]) - 1):
        for s in range(len(env.discrete["speed"]) - 1):
            for c_p in range(len(env.discrete["c-pos"]) - 1):
                for c_s in range(len(env.discrete["c-speed"]) - 1):
                    action_sum = [0] * 3
                    for angle, speed, c_pos, c_speed in itertools.product(
                        (env.discrete["energy"][e], env.discrete["energy"][e + 1]),
                        (env.discrete["speed"][s], env.discrete["speed"][s + 1]),
                        (env.discrete["c-pos"][c_p], env.discrete["c-pos"][c_p + 1]),
                        (env.discrete["c-speed"][c_s], env.discrete["c-speed"][c_s + 1]),
                    ):
                        reward_max = float("-inf")
                        for action in range(3):
                            env.set_state(c_pos, c_speed, angle, speed)
                            _obs, reward, _term, _trunc, _ = env.step(action)
                            if reward > reward_max:
                                action_max = action
                                reward_max = reward
                        action_sum[action_max] += 1
                    if (
                        max(action_sum) != 16
                        and action_sum[0] > 0
                        and action_sum[2] > 0
                        and action_sum[0] == action_sum[2]
                    ):
                        logger.info(f"angle:{e}, speed:{s}, c_pos:{c_p}, c_speed:{c_s}: {action_sum}")


@pytest.mark.skip(reason="Test must be updated")
def test_state(crane: Callable[..., Crane], *, show: bool) -> None:  # noqa: PLR0915
    """Set state and calculate reward."""

    def check_step(
        act: int,
        *,
        obs: tuple[int, ...] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> None:
        _obs, _reward, _terminated, _truncated, _ = env.step(1)
        if obs is not None:
            assert np.allclose(_obs, obs), f"obs. Found {_obs}. Expected {obs}"
        if reward is not None:
            assert abs(reward - _reward) < 1e-9, f"reward. Found {_reward}. Expected {reward}"
        assert terminated is None or _terminated == terminated, f"terminated. Found {_terminated}.Expected {terminated}"
        assert truncated is None or _truncated == truncated, f"truncated. Found {_truncated}. Expected {truncated}"

    def get_state():
        """Get state variables as tuple and text."""
        assert isinstance(env.wire, Wire)
        state = (
            float(env.crane.position[0]),
            float(env.crane.velocity[0]),
            float(np.degrees(np.pi - env.wire.boom[1])),
            float(env.wire.cm_v[0]),
        )
        txt = f"pos:{state[0]}, speed:{state[1]}, angle:{state[2]}, x-speed:{state[3]}"
        return state, txt

    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=1.0,
            render_mode="none",
            reward_limit=0.0,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.01}),
            discrete="energy",
        ),
    )
    assert isinstance(env.wire, Wire)
    env.reset()
    agent = QLearningAgent(env)

    env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.0)
    for _i in range(10):
        assert np.allclose(get_state()[0], (0, 0, 0, 0))
        env.step(1)  # check that nothing moves

    env.step(0)
    _state = get_state()[0]
    assert np.allclose(get_state()[0][:2], (-0.1, -0.1)), f"Found {get_state()[0][:2]}"
    assert get_state()[0][2] > 0.26
    assert get_state()[0][3] > 0.08

    # env.reset()
    env.set_state(1.0, 2.0, 0.0, 0.0)
    assert np.allclose(get_state()[0], (1.0, 2.0, 0, 0)), f"Found {get_state()[0]}"
    env.step(1)
    assert np.allclose(get_state()[0], (3.0, 2.0, 5.532267250195097, -1.7318224679112262)), f"Found {get_state()[0]}"

    env.set_state(0.0, 0.0, np.radians(10), -2.0)
    assert np.allclose(get_state()[0], (0.0, 0.0, 10, -2)), f"Found {get_state()[0]}"
    env.step(1)
    assert np.allclose(get_state()[0], (0.0, 0.0, 1.0274015154955907, -0.8379044477976203)), f"Found {get_state()[0]}"
    # logger.info(f"State: {get_state()[1]}")
    # check_step( 1, obs=(0, 0, 0, 0, 0, 0, 0, 1, 1), reward=0, terminated=True, truncated=False)
    # env.set_state(pos=18.0, speed=0.0, direction=0.0, w_speed=0.0)
    # logger.info( env.step(1))
    env.set_state(pos=2.0, speed=0.0, direction=0.0, w_speed=0.0)
    res = env.step(1)  # neutral step
    actions = (
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        1,
    )
    reward = float("-inf")
    reward_sum = 0.0
    agent.episodes_init()
    for s in range(len(actions)):
        obs0, _reward0 = res[:2]
        res = env.step(actions[s])
        if res[1] <= reward:
            logger.info(f"step:{s}, actions:{actions[:s]}, obs:{res[0]}, reward:{float(res[1])}")
            break
        agent.update_q(obs0, actions[s], res[1], terminated=False, next_obs=res[0], _prev_reward=reward)
        reward = float(res[1])
        reward_sum += reward
    logger.info(f"reward:{reward}, avg:{reward_sum / s}")
    logger.info(f"pos:{env.crane.position}, speed:{env.crane.velocity}, dir:{env.wire.direction}, v_w:{env.wire.cm_v}")
    for k, v in agent.q_values.items():
        logger.info(k, v)
    # env.set_state(pos=18.0, speed=0.0, direction=0.0, w_speed=0.0)
    # logger.info( env.step(2))


@pytest.mark.skip(reason="Test must be updated")
def test_state2(crane: Callable[..., Crane], *, show: bool) -> None:
    """Set state and calculate reward."""

    def check_step(
        act: int,
        *,
        obs: tuple[int, ...] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> None:
        _obs, _reward, _terminated, _truncated, _ = env.step(1)
        if obs is not None:
            assert np.allclose(_obs, obs), f"obs. Found {_obs}. Expected {obs}"
        if reward is not None:
            assert abs(reward - _reward) < 1e-9, f"reward. Found {_reward}. Expected {reward}"
        assert terminated is None or _terminated == terminated, f"terminated. Found {_terminated}.Expected {terminated}"
        assert truncated is None or _truncated == truncated, f"truncated. Found {_truncated}. Expected {truncated}"

    def get_state():
        """Get state variables as tuple and text."""
        assert isinstance(env.wire, Wire)
        state = (
            float(env.crane.position[0]),
            float(env.crane.velocity[0]),
            float(np.degrees(np.pi - env.wire.boom[1])),
            float(env.wire.cm_v[0]),
        )
        txt = f"pos:{state[0]}, speed:{state[1]}, angle:{state[2]}, x-speed:{state[3]}"
        return state, txt

    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=1.0,
            render_mode="none",
            reward_limit=0.0,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.0}),
            discrete="phase",
        ),
    )
    env.reset()
    _agent = QLearningAgent(env)

    env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.0)
    for _i in range(10):
        assert np.allclose(get_state()[0], (0, 0, 0, 0))
        env.step(1)  # check that nothing moves

    env.step(0)
    state = get_state()[0]
    assert np.allclose(state, (-0.1, -0.1, 0.26384339641900634, 0.08276160999714069))
    logger.info("Experiment:")
    for a in range(3):
        env.set_state(pos=0.0, speed=0.0, direction=np.radians(3.0), w_speed=1.5)
        env.step(a)
        # env.step(1)
        logger.info(a, get_state()[1], env._get_obs())


@pytest.mark.skip(reason="Test must be updated")
def test_update_q_values(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="none",
            reward_limit=-0.05,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.01}),
            discrete="energy",
        ),
    )
    env.reset()
    agent = QLearningAgent(env)
    env.set_state(pos=2.0, speed=0.0, direction=0.0, w_speed=0.0)
    env.step(1)  # neutral step
    agent.episodes_init()

    obs, _ = env.reset()  # first reward is also available as self.env.reward
    # num_failed = 0

    for _i in range(1000):
        prev_reward = env.reward
        action = agent.get_action(obs)  # choose action (initially random, gradually more intelligent)
        next_obs, _reward, _terminated, _truncated, _ = env.step(action)  # take action and observe result
        reward = float(_reward)
        agent.update_q(obs, action, reward, terminated=False, next_obs=next_obs, _prev_reward=prev_reward)
        # Move to next state
        obs = next_obs
        # truncated = False

    logger.info(f"REWARDS: {env.rewards}")


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_levels(build_crane)
    # test_smoke(build_crane, show=True)
    # test_q_analyse(build_crane, show=True)
    # test_intervals(build_crane)
    # test_state(build_crane, show=True)
    # test_state2(build_crane, show=True)
    # test_update_q_values(build_crane, show=True)
    # test_discretization(build_crane, show=True, discretization='energy')
