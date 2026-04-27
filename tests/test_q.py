import logging
from collections.abc import Callable

import numpy as np
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


def test_smoke(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        start_speed=-1.0,
        render_mode="plot" if show else "none",
        reward_limit=-0.05,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = QLearningAgent(env, trained=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


def test_q_analyse(crane: Callable[..., Crane], *, trained: tuple[str, bool]|None) -> None:
    assert trained is not None, "Cannot analyse q-values if no pre-trained data are supplied"
    env = AntiPendulumEnv(
        crane,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = QLearningAgent(env, trained=trained)
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


def test_intervals(crane: Callable[..., Crane]):
    """Test that learning / saving / resuming learning works:"""
    save_path = Path.cwd() / "q_interval_1.json"
    env = AntiPendulumEnv(
        crane,
        start_speed=-1.0,
        render_mode="none",
        reward_limit=-0.05,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    
    agent = QLearningAgent(env, trained=(save_path, False))
    for i in range(10):
        _ = env.reset(seed=i + 1)
        agent.do_episodes(n_episodes=10)
        if i == 0:
            agent = QLearningAgent(env, trained=(save_path, True))
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = 0#pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_smoke(build_crane, show=True)
    # test_q_analyse(build_crane, trained=None)
    test_intervals(build_crane)

