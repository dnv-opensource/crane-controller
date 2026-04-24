import logging
from collections.abc import Callable

import pytest
from py_crane.crane import Crane

from crane_controller.algorithm import AlgorithmAgent
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_algorithm_strategies(
    crane: Callable[..., Crane],
    *,
    show: bool,
) -> None:
    env = AntiPendulumEnv(
        crane,
        start_speed=0.0,
        render_mode="plot" if show else "none",
        reward_limit=1000.0,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = AlgorithmAgent(env)
    agent.do_strategies(max_steps=5000 if show else 10)


def test_algorithm(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        start_speed=0.0,
        seed=1,
        render_mode="plot" if show else "none",
        reward_limit=1000.0,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = AlgorithmAgent(env)
    agent.strategy = (0, 2, 0, 2)
    logger.info("Best strategy (0,2,0,2) in start mode")
    agent.do_episodes(1, max_steps=5000)
    agent.strategy = (1, 1, 1, 1)
    logger.info("Do-nothing strategy (1,1,1,1) in start mode")
    agent.do_episodes(1, max_steps=5000)
    env.start_speed = 1.0
    agent.strategy = (2, 1, 1, 0)
    logger.info("Best strategy (2,1,1,0) in stop mode")
    agent.do_episodes(1, max_steps=5000)
    agent.strategy = (1, 1, 1, 1)
    logger.info("Do-nothing strategy (1,1,1,1) in stop mode")
    agent.do_episodes(1, max_steps=5000)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_algorithm_strategies(build_crane, show=True, reward_limit=1000.0, start_speed=0.0)
    # test_algorithm(build_crane, show=True)
