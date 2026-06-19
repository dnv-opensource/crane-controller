import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from crane_controller.envs.simple_test_env import Config, SimpleTestEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


def test_env():
    env = SimpleTestEnv(
        config=Config(
            acc=1.0,
            pos_range=(-100, 100),
            speed_range=(-10, 10),
            pos0=0.0,
            speed0=0.0,
            pos1=10.0,
            speed1=0.0,
            seed=1,
        ),
        reward_fac=(1.0, 1.0),
        reward_limit=1000,
        dt=1.0,
    )
    assert env.config is not None
    assert env.action_space.n == 3  # type: ignore[attr-defined]  ## the attribute exists
    pos = env.pos
    speed = env.speed
    dt = env.dt
    for _i in range(1000):
        i_acc = int(env.action_space.sample())  # cast np.uint16 → int to avoid underflow
        a = (i_acc - 1) * env.config.acc  # action 0→-acc, 1→0, 2→+acc
        obs, _reward, _term, _trunc, _ = env.step(i_acc)
        pos += speed * dt + 0.5 * a * dt * dt
        speed += a * dt
        assert pos == env.pos
        assert round(pos) == obs[0]
        assert speed == env.speed
        assert round(speed) == obs[1]


def test_smoke(*, show: bool) -> None:
    env = SimpleTestEnv(
        config=Config(
            acc=1.0,
            pos_range=(-100, 100),
            speed_range=(-10, 10),
            pos0=0.0,
            speed0=0.0,
            pos1=10.0,
            speed1=0.0,
        ),
        reward_fac=(1.0, 1.0),
        reward_limit=None,
        dt=1.0,
        render_mode="plot",
    )
    agent = QLearningAgent(env, filename=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


@pytest.mark.skip(reason="needs 'env' fixture and pre-trained q_trained.json; run manually via __main__")
def test_q_analyse(env: gym.Env[tuple[int, ...] | np.ndarray, int], *, show: bool) -> None:
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


if __name__ == "__main__":
    import os
    from pathlib import Path

    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_env()
    # test_smoke(show=True)
    # env = SimpleTestEnv(config=None, reward_fac=(1.0, 1.0), reward_limit=None, dt=1.0)
    # test_q_analyse(env, show=True)
