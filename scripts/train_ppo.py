"""Train a PPO agent on the AntiPendulumEnv.

Examples:
--------
.. code-block:: bash

    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --config experiments/baseline.yaml
    uv run python scripts/train_ppo.py --config experiments/baseline.yaml --steps 500000
    uv run python scripts/train_ppo.py --reward-fac 1.0 0.0015 0.0 0.005 0.01
    uv run python scripts/train_ppo.py --steps 500000 --n-envs 8 --save-path models/ppo.zip
    uv run python scripts/train_ppo.py --resume-from models/ppo_AntiPendulumEnv.zip --steps 50000
    uv run python scripts/train_ppo.py --dry-run
"""

import argparse
import logging
from pathlib import Path

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv
from crane_controller.experiment_config import (
    ExperimentConfig,
    RewardConfig,
    TrainingConfig,
    load_experiment_config,
    load_training_sidecar,
    save_training_sidecar,
)
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

LOGGER = logging.getLogger(__name__)


def main() -> None:  # noqa: PLR0915
    """Parse CLI arguments and train a PPO agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Pre-parse --config before the main parser so YAML values can seed defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    _ = pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_experiment_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train a PPO agent on the crane anti-pendulum task.")
    _ = parser.add_argument(
        "--steps",
        type=int,
        default=config.training.steps,
        help="Timesteps for this run (default from --config or 100 000). Does not cap total across resumes.",
    )
    _ = parser.add_argument(
        "--n-envs", type=int, default=config.training.n_envs, help="Number of parallel environments"
    )
    _ = parser.add_argument("--render-mode", type=str, default="none", help="Render mode during training")
    _ = parser.add_argument(
        "--save-path",
        type=str,
        default=config.training.save_path,
        help="Where to save the trained model",
    )
    _ = parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a saved model zip to resume training from.",
    )
    _ = parser.add_argument(
        "--gamma",
        type=float,
        default=config.training.gamma,
        help="Discount factor for future rewards (default 0.99). Try 0.999 for longer planning horizon.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1000 steps with live reward-tracking plot, without saving the model.",
    )
    _ = parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML experiment config file.",
    )
    _ = parser.add_argument(
        "--reward-fac",
        type=float,
        nargs=5,
        default=None,
        metavar=("ENERGY", "POSITIONAL", "TIME", "POSITION", "ACCELERATION"),
        help="Override all five reward weights (beats --config).",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for PPO initialisation. Omit for non-deterministic training.",
    )
    _ = parser.add_argument(
        "--ent-coef",
        type=float,
        default=config.training.ent_coef,
        help="Entropy bonus coefficient (default 0.0). Try 0.005-0.01 to reduce seed sensitivity.",
    )
    _ = parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.training.learning_rate,
        help="Adam learning rate (default 3e-4).",
    )
    _ = parser.add_argument(
        "--clip-range",
        type=float,
        default=config.training.clip_range,
        help="PPO clipping parameter (default 0.2). Lower = more conservative updates.",
    )
    _ = parser.add_argument(
        "--n-steps",
        type=int,
        default=config.training.n_steps,
        help="Timesteps per env before each gradient update (default 2048). Try 8192 for more stable gradients.",
    )
    _ = parser.add_argument(
        "--randomize-start",
        action="store_true",
        default=config.training.randomize_start,
        help="Randomise initial pendulum speed each episode (default False).",
    )
    _ = parser.add_argument(
        "--rail-limit",
        type=float,
        default=config.training.rail_limit,
        help="Half-span of the crane rail in metres (default 10.0). "
        "Reduce to e.g. 2.0 for earlier truncation and tighter credit assignment.",
    )
    _ = parser.add_argument(
        "--start-speed",
        type=float,
        default=config.training.start_speed,
        help="Initial pendulum speed (default 1.0). Upper bound of training range when --randomize-start is set.",
    )
    _ = parser.add_argument(
        "--continuous-actions",
        "--no-continuous-actions",
        action=argparse.BooleanOptionalAction,
        default=config.training.continuous_actions,
        help="Use Box(-1,1) action space for PPO (default True). Pass --no-continuous-actions for Discrete(3).",
    )
    _ = parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=config.training.max_episode_steps,
        help="TimeLimit cap per episode (default 1000).",
    )
    args = parser.parse_args()

    # Resolve final reward config: explicit --reward-fac beats loaded YAML/defaults.
    reward_config = RewardConfig(*args.reward_fac) if args.reward_fac is not None else config.reward

    experiment_config = ExperimentConfig(
        reward=reward_config,
        training=TrainingConfig(
            steps=args.steps,
            n_envs=args.n_envs,
            gamma=args.gamma,
            save_path=args.save_path,
            seed=args.seed,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            n_steps=args.n_steps,
            randomize_start=args.randomize_start,
            rail_limit=args.rail_limit,
            start_speed=args.start_speed,
            continuous_actions=args.continuous_actions,
            max_episode_steps=args.max_episode_steps,
        ),
        config_source=pre_args.config,
    )

    if args.dry_run:
        agent = ProximalPolicyOptimizationAgent(
            AntiPendulumEnv,
            n_envs=1,
            env_kwargs={
                "crane": build_crane,
                "conf": AntiPendulumConfig(start_speed=-1.0, render_mode="reward-tracking"),
            },
        )
        agent.do_training(1000, progress_bar=False)
    elif args.resume_from:
        # Warn when --steps overrides a YAML value on resume — the cap is not enforced.
        if pre_args.config is not None and args.steps != config.training.steps:
            LOGGER.warning(
                "--steps %d overrides the experiment config value (%d). "
                "Training will run %d steps from the checkpoint — not a cumulative cap.",
                args.steps,
                config.training.steps,
                args.steps,
            )

        # Priority: --reward-fac > --config > sidecar from checkpoint > defaults
        if args.reward_fac is not None:
            resume_reward = RewardConfig(*args.reward_fac)
        elif pre_args.config is not None:
            resume_reward = config.reward
        else:
            try:
                resume_reward = load_training_sidecar(args.resume_from).reward
            except FileNotFoundError:
                LOGGER.warning("No sidecar found for %s; using config defaults.", args.resume_from)
                resume_reward = config.reward
        resume_config = ExperimentConfig(
            reward=resume_reward,
            training=TrainingConfig(
                steps=args.steps,
                n_envs=args.n_envs,
                gamma=args.gamma,
                save_path=args.save_path,
                continuous_actions=args.continuous_actions,
            ),
            config_source=pre_args.config,
        )
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent = ProximalPolicyOptimizationAgent.resume(
            AntiPendulumEnv,
            model_path=args.resume_from,
            env_kwargs={
                "crane": build_crane,
                "conf": AntiPendulumConfig(
                    start_speed=args.start_speed,
                    randomize_start=args.randomize_start,
                    render_mode=args.render_mode,
                    reward_fac=resume_config.reward,
                    rail_limit=args.rail_limit,
                    reward_limit=resume_config.training.reward_limit,
                    continuous_actions=args.continuous_actions,
                ),
            },
            save_path=args.save_path,
            n_envs=args.n_envs,
            max_episode_steps=resume_config.training.max_episode_steps,
        )
        csv_path = str(Path(args.save_path).with_name(Path(args.save_path).stem + "_log.csv"))
        agent.do_training(args.steps, reset_num_timesteps=False, csv_path=csv_path)
        _ = save_training_sidecar(args.save_path, resume_config)
        vecnorm_path = Path(args.save_path).parent / f"{Path(args.save_path).stem}_vecnorm.pkl"
        LOGGER.info("Model saved to %s", args.save_path)
        LOGGER.info("VecNormalize stats saved to %s", vecnorm_path)
    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent = ProximalPolicyOptimizationAgent(
            AntiPendulumEnv,
            n_envs=args.n_envs,
            env_kwargs={
                "crane": build_crane,
                "conf": AntiPendulumConfig(
                    start_speed=args.start_speed,
                    randomize_start=args.randomize_start,
                    render_mode=args.render_mode,
                    reward_fac=experiment_config.reward,
                    rail_limit=experiment_config.training.rail_limit,
                    reward_limit=experiment_config.training.reward_limit,
                    continuous_actions=args.continuous_actions,
                ),
            },
            save_path=args.save_path,
            gamma=args.gamma,
            seed=args.seed,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            n_steps=args.n_steps,
            max_episode_steps=experiment_config.training.max_episode_steps,
        )
        csv_path = str(Path(args.save_path).with_name(Path(args.save_path).stem + "_log.csv"))
        agent.do_training(args.steps, csv_path=csv_path)
        _ = save_training_sidecar(args.save_path, experiment_config)
        vecnorm_path = Path(args.save_path).parent / f"{Path(args.save_path).stem}_vecnorm.pkl"
        LOGGER.info("Model saved to %s", args.save_path)
        LOGGER.info("VecNormalize stats saved to %s", vecnorm_path)


if __name__ == "__main__":
    main()
