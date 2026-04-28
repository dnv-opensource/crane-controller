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
        save_path=save_path,
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


def test_ppo_inference_disables_training_mode(crane: Callable[..., Crane], tmp_path: Path) -> None:
    """Test that load() sets VecNormalize to evaluation mode."""
    save_path = str(tmp_path / "model.zip")
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        save_path=save_path,
    )
    agent.do_training(500, progress_bar=False)

    loaded = ProximalPolicyOptimizationAgent.load(
        AntiPendulumEnv,
        model_path=save_path,
        env_kwargs={"crane": crane, "start_speed": 1.0},
    )
    assert not loaded.vec_env.training
    assert not loaded.vec_env.norm_reward


def test_ppo_resume_keeps_training_mode(crane: Callable[..., Crane], tmp_path: Path) -> None:
    """Test that resume() keeps VecNormalize in training mode."""
    save_path = str(tmp_path / "model.zip")
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        save_path=save_path,
    )
    agent.do_training(500, progress_bar=False)

    resumed = ProximalPolicyOptimizationAgent.resume(
        AntiPendulumEnv,
        model_path=save_path,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        n_envs=1,
    )
    assert resumed.vec_env.training
    assert resumed.vec_env.norm_reward


def test_ppo_resume_updates_vecnorm(crane: Callable[..., Crane], tmp_path: Path) -> None:
    """Test that VecNormalize statistics update during resumed training."""
    save_path = str(tmp_path / "model.zip")
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,
        n_envs=1,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        save_path=save_path,
    )
    agent.do_training(500, progress_bar=False)

    resumed = ProximalPolicyOptimizationAgent.resume(
        AntiPendulumEnv,
        model_path=save_path,
        env_kwargs={"crane": crane, "start_speed": 1.0},
        save_path=save_path,
        n_envs=1,
    )
    mean_before = resumed.vec_env.obs_rms.mean.copy()  # type: ignore[attr-defined]
    resumed.do_training(500, progress_bar=False, reset_num_timesteps=False)
    assert not np.allclose(resumed.vec_env.obs_rms.mean, mean_before)  # type: ignore[attr-defined]
