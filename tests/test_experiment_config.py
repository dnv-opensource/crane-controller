"""Tests for crane_controller.experiment_config."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from crane_controller.experiment_config import (
    ExperimentConfig,
    RewardConfig,
    TrainingConfig,
    _meta_path,  # type: ignore[reportPrivateUsage]
    load_experiment_config,
    load_training_sidecar,
    save_training_sidecar,
)

# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------


def test_reward_config_defaults() -> None:
    rc = RewardConfig()
    assert rc.energy == 1.0
    assert rc.positional == 0.0015
    assert rc.time == 0.0
    assert rc.position == 0.005
    assert rc.acceleration == 0.01
    assert rc.terminal_penalty == 0.0
    assert rc.angle == 0.0
    assert rc.angular_velocity == 0.0
    assert rc.crane_velocity == 0.0
    assert rc.crane_acceleration == 0.0
    assert rc.angular_acceleration == 0.0


def test_reward_config_from_dict_full() -> None:
    d = {
        "energy": 2.0,
        "positional": 0.001,
        "time": 0.01,
        "position": 0.1,
        "acceleration": 0.05,
        "terminal_penalty": -5.0,
    }
    rc = RewardConfig.from_dict(d)
    assert rc == RewardConfig(
        energy=2.0, positional=0.001, time=0.01, position=0.1, acceleration=0.05, terminal_penalty=-5.0
    )


def test_reward_config_from_dict_partial_fills_defaults() -> None:
    rc = RewardConfig.from_dict({"energy": 5.0})
    assert rc.energy == 5.0
    assert rc.positional == RewardConfig().positional
    assert rc.acceleration == RewardConfig().acceleration


def test_reward_config_from_dict_int_values_coerced_to_float() -> None:
    rc = RewardConfig.from_dict({"energy": 1, "position": 0})
    assert isinstance(rc.energy, float)
    assert isinstance(rc.position, float)


def test_reward_config_is_frozen() -> None:
    rc = RewardConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rc.energy = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_defaults() -> None:
    tc = TrainingConfig()
    assert tc.steps == 100_000
    assert tc.n_envs == 4
    assert tc.gamma == 0.99
    assert tc.save_path == "models/ppo_AntiPendulumEnv.zip"
    assert tc.seed is None
    assert tc.ent_coef == 0.0
    assert tc.learning_rate == 3e-4
    assert tc.clip_range == 0.2
    assert tc.n_steps == 2048
    assert tc.randomize_start is False
    assert tc.rail_limit == 10.0
    assert tc.continuous_actions is True


def test_training_config_from_dict() -> None:
    d = {
        "steps": 500000,
        "n_envs": 8,
        "gamma": 0.999,
        "save_path": "models/my_model.zip",
        "seed": 42,
        "ent_coef": 0.01,
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "n_steps": 8192,
        "randomize_start": True,
    }
    tc = TrainingConfig.from_dict(d)
    assert tc == TrainingConfig(
        steps=500000,
        n_envs=8,
        gamma=0.999,
        save_path="models/my_model.zip",
        seed=42,
        ent_coef=0.01,
        learning_rate=1e-4,
        clip_range=0.1,
        n_steps=8192,
        randomize_start=True,
    )


def test_training_config_from_dict_partial_fills_defaults() -> None:
    tc = TrainingConfig.from_dict({"steps": 1000})
    assert tc.steps == 1000
    assert tc.n_envs == TrainingConfig().n_envs


# ---------------------------------------------------------------------------
# load_experiment_config
# ---------------------------------------------------------------------------


def test_load_experiment_config_none_returns_defaults() -> None:
    ec = load_experiment_config(None)
    assert ec.reward == RewardConfig()
    assert ec.training == TrainingConfig()
    assert ec.config_source is None


def test_load_experiment_config_reads_yaml(tmp_path: Path) -> None:
    yaml_content = "reward:\n  energy: 2.0\n  position: 0.1\ntraining:\n  steps: 50000\n  gamma: 0.95\n"
    cfg_file = tmp_path / "test.yaml"
    _ = cfg_file.write_text(yaml_content)
    ec = load_experiment_config(cfg_file)
    assert ec.reward.energy == 2.0
    assert ec.reward.position == 0.1
    assert ec.training.steps == 50000
    assert ec.training.gamma == 0.95
    assert ec.config_source == str(cfg_file)


def test_load_experiment_config_nonexistent_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_experiment_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# save / load sidecar
# ---------------------------------------------------------------------------


def test_meta_path_convention() -> None:
    assert _meta_path("models/ppo_foo.zip") == Path("models/ppo_foo_meta.json")


def test_save_training_sidecar_creates_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model.zip"
    config = ExperimentConfig()
    sidecar = save_training_sidecar(model_path, config)
    assert sidecar.exists()
    assert sidecar.suffix == ".json"


def test_save_training_sidecar_content(tmp_path: Path) -> None:
    model_path = tmp_path / "model.zip"
    config = ExperimentConfig(
        reward=RewardConfig(energy=2.0),
        training=TrainingConfig(steps=999),
    )
    sidecar = save_training_sidecar(model_path, config)
    payload = json.loads(sidecar.read_text())
    assert payload["reward"]["energy"] == 2.0
    assert payload["training"]["steps"] == 999


def test_load_training_sidecar_round_trip(tmp_path: Path) -> None:
    model_path = tmp_path / "model.zip"
    original = ExperimentConfig(
        reward=RewardConfig(position=0.1, acceleration=0.05),
        training=TrainingConfig(steps=250000, gamma=0.999),
    )
    _ = save_training_sidecar(model_path, original)
    restored = load_training_sidecar(model_path)
    assert restored.reward == original.reward
    assert restored.training == original.training


def test_load_training_sidecar_missing_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "model.zip"
    with pytest.raises(FileNotFoundError):
        _ = load_training_sidecar(model_path)
