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
    agent.do_strategies()


def test_algorithm(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        start_speed=0.0,
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
