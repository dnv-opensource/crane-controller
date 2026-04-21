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


def test_smoke(crane: Callable):
    env = AntiPendulumEnv(crane, start_speed=-1.0, render_mode="none", reward_limit=-0.05, discrete=_DISCRETE.copy())
    agent = QLearningAgent(env, trained=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


def test_q_analyse(crane: Callable, trained: tuple[str, bool] = ("anti-pendulum.json", False)):
    env = AntiPendulumEnv(
        crane,
        discrete=_DISCRETE.copy(),
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
