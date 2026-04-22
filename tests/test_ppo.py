import logging

from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

logger = logging.getLogger(__name__)


def test_monitor(crane: Crane, *, show: bool) -> None:
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type]
        n_envs=1,
        env_kwargs={
            "crane": crane,
            "seed": 2,
            "start_speed": 1.0,
            "render_mode": "reward-tracking" if show else "none",
        },
    )
    agent.do_training(1000)
