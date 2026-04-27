import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from py_crane.crane import Crane
from stable_baselines3.common.running_mean_std import RunningMeanStd

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

logger = logging.getLogger(__name__)


def test_monitor(crane: Callable[..., Crane], *, show: bool) -> None:
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={
            "crane": crane,
            "seed": 2,
            "start_speed": 1.0,
            "render_mode": "reward-tracking" if show else "none",
        },
    )
    agent.do_training(1000)


def test_ppo_saves_vecnorm(crane: Callable[..., Crane], tmp_path: Path) -> None:
    """Test that do_training saves the VecNormalize statistics alongside the model."""
    save_path = str(tmp_path / "model.zip")
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        trained=(save_path, True),
    )
    agent.do_training(500, progress_bar=False)
    assert (tmp_path / "model_vecnorm.pkl").exists()


def test_ppo_vecnorm_updates(crane: Callable[..., Crane]) -> None:
    """Test that the VecNormalize running mean is updated during training."""
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={"crane": crane, "start_speed": 1.0},
    )
    agent.do_training(500, progress_bar=False)
    assert isinstance(agent.vec_env.obs_rms, RunningMeanStd)
    assert not np.allclose(agent.vec_env.obs_rms.mean, 0.0)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest # noqa: F401
    from crane_controller.crane_factory import build_crane  # noqa: F401

    retcode = pytest.main(["-rP -s -v", "--show", "False", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_monitor(build_crane, show=True)
