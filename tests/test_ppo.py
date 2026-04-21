import logging

from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

logger = logging.getLogger(__name__)


def test_monitor(crane: Crane):
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type]
        n_envs=1,
        env_kwargs={
            "crane": crane,
            "seed": 2,
            "start_speed": 1.0,
            "render_mode": "none",
        },
    )
    agent.do_training(1000)


# TODO: move to scripts/train_ppo.py — not a unit test (no assertions, too slow for CI)
# def test_training_ppo(crane, n_envs=4, nsteps=100000, render_mode="data", trained=None): ...

# TODO: move to scripts/run_ppo.py — not a unit test (no assertions, requires external model file)
# def test_act_ppo(crane, render_mode="play_back", trained=("ppo_AntiPendulumEnv.zip", False), episodes=1): ...
