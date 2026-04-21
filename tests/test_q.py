import logging
from typing import Callable

import numpy as np
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)

_DISCRETE = {
    "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
    "pos": (0, 1),
    "speed": (0, 1),
    "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "sector": (0, 1),
}


def test_interval_training_q(crane: Crane, intervals=10, render_mode: str = "none", reward_limit=0.01):
    env = AntiPendulumEnv(
        crane,
        start_speed=-1.0,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "direction": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = QLearningAgent(env, trained=("interval.json", False))
    for i in range(intervals):
        logger.info(f"Start interval {i}...")
        env.reset(seed=i + 1)
        agent.do_episodes(n_episodes=10)
        if i == 0:
            agent = QLearningAgent(env, trained=("interval.json", True))


def test_training_q(
    crane: Callable,
    episodes: int = 10000,
    render_mode: str = "none",
    reward_limit=-0.05,
    trained=None,
    v0: float = -1.0,
    max_steps: int = 1000,
    show: int = 0,
):
    env = AntiPendulumEnv(
        crane,
        start_speed=v0,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete=_DISCRETE,
    )
    agent = QLearningAgent(env, trained=trained)
    logger.info(f"Agent {agent.env} initialized. Start training...")
    agent.do_episodes(n_episodes=episodes, max_steps=max_steps, show=show)
    logger.info(f"Training done. Resets:{agent.env.nresets}, Successes:{agent.env.nsuccess}")  # type: ignore


def test_q_analyse(crane, trained: tuple[str, bool] = ("anti-pendulum.json", False)):
    env = AntiPendulumEnv(
        crane,
        discrete=_DISCRETE,
    )
    agent = QLearningAgent(env, trained=trained)
    for k, v in agent.q_values.items():
        assert len(k) == 5, len(v) == 3
    for pos in (0, 1):
        for speed in (0, 1):
            res = dict((k, v) for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed)
            print(f"pos:{pos}, speed:{speed}")
            acc = []
            for i in range(3):
                col = [x[i] for x in res.values()]
                acc.append(np.average(col))
            print(acc)
