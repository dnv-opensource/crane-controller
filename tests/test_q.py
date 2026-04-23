import logging
from typing import Callable

import numpy as np

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logger = logging.getLogger(__name__)


def test_smoke(crane: Callable, show: bool):
    env = AntiPendulumEnv(
        crane,
        start_speed=-1.0,
        render_mode="plot" if show else "none",
        reward_limit=-0.05,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = QLearningAgent(env, trained=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


def test_q_analyse(crane: Callable, trained: tuple[str, bool] = ("anti-pendulum.json", False)):
    env = AntiPendulumEnv(
        crane,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
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


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_smoke(build_crane, show=True)
    # test_q_analyse(build_crane)
