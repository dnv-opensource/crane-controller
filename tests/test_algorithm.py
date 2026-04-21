import logging
from typing import Callable

from crane_controller.algorithm import AlgorithmAgent
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv

logger = logging.getLogger(__name__)

_DISCRETE = {
    "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
    "pos": (0, 1),
    "speed": (0, 1),
    "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "sector": (0, 1),
}


def test_algorithm_strategies(
    crane: Callable,
    render_mode: str = "plot",
    reward_limit: float = 1000.0,
    start_speed: float = 0.0,
):
    env = AntiPendulumEnv(
        crane,
        start_speed=start_speed,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete=_DISCRETE,
    )
    agent = AlgorithmAgent(env)
    agent.do_strategies()


def test_algorithm(crane: Callable, render_mode: str = "plot"):
    env = AntiPendulumEnv(
        crane,
        start_speed=0.0,
        render_mode=render_mode,
        reward_limit=1000.0,
        discrete=_DISCRETE,
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
