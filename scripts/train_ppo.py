"""Train a PPO agent on the AntiPendulumEnv.

Examples
--------
.. code-block:: bash

    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --steps 500000 --n-envs 8 --save-path models/ppo.zip
    uv run python scripts/train_ppo.py --resume-from models/ppo_AntiPendulumEnv.zip --steps 50000
    uv run python scripts/train_ppo.py --dry-run
"""

import argparse
import logging
from pathlib import Path

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and train a PPO agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train a PPO agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    _ = parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    _ = parser.add_argument("--render-mode", type=str, default="none", help="Render mode during training")
    _ = parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_AntiPendulumEnv.zip",
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
        default=0.99,
        help="Discount factor for future rewards (default 0.99). Try 0.999 for longer planning horizon.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1000 steps with live reward-tracking plot, without saving the model.",
    )
    args = parser.parse_args()

    if args.dry_run:
        agent = ProximalPolicyOptimizationAgent(
            AntiPendulumEnv,
            n_envs=1,
            env_kwargs={
                "crane": build_crane,
                "start_speed": -1.0,
                "render_mode": "reward-tracking",
            },
        )
        agent.do_training(1000, progress_bar=False)
    elif args.resume_from:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent = ProximalPolicyOptimizationAgent.resume(
            AntiPendulumEnv,
            model_path=args.resume_from,
            env_kwargs={
                "crane": build_crane,
                "start_speed": 1.0,
                "render_mode": args.render_mode,
                "reward_fac": (1.0, 0.0015, 0.0),
            },
            save_path=args.save_path,
            n_envs=args.n_envs,
        )
        agent.do_training(args.steps, reset_num_timesteps=False)
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
                "start_speed": 1.0,
                "render_mode": args.render_mode,
                "reward_fac": (1.0, 0.0015, 0.0),
            },
            save_path=args.save_path,
            gamma=args.gamma,
        )
        agent.do_training(args.steps)
        vecnorm_path = Path(args.save_path).parent / f"{Path(args.save_path).stem}_vecnorm.pkl"
        LOGGER.info("Model saved to %s", args.save_path)
        LOGGER.info("VecNormalize stats saved to %s", vecnorm_path)


if __name__ == "__main__":
    main()
