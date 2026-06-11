"""Run a trained PPO agent on the AntiPendulumEnv.

Examples
--------
.. code-block:: bash

    uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip
    uv run python scripts/play_ppo.py --model-path models/ppo.zip --render-mode plot --episodes 3
"""

import argparse
import logging
from pathlib import Path

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.experiment_config import load_training_sidecar
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and run a trained PPO agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Pre-parse --model-path so the sidecar can seed argument defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    _ = pre_parser.add_argument("--model-path", type=str, required=True)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_training_sidecar(pre_args.model_path)

    parser = argparse.ArgumentParser(description="Run a trained PPO agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--model-path", type=str, required=True, help="Path to a trained .zip model")
    _ = parser.add_argument("--render-mode", type=str, default="play-back", help="Render mode for playback")
    _ = parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    _ = parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=config.training.randomize_start,
        help="Randomise initial pendulum speed each episode (default from model sidecar).",
    )
    _ = parser.add_argument(
        "--start-speed",
        type=float,
        default=1.0,
        help="Initial pendulum speed for playback (default 1.0).",
    )
    _ = parser.add_argument(
        "--continuous-actions",
        "--no-continuous-actions",
        action=argparse.BooleanOptionalAction,
        default=config.training.continuous_actions,
        help="Use Box(-1,1) action space (default from model sidecar).",
    )
    _ = parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Override TimeLimit for play episodes (default: value from model sidecar). "
            "Pass 3000 when playing old pre-trained models that have no max_episode_steps "
            "in their sidecar (those default to 100 otherwise)."
        ),
    )
    _ = parser.add_argument(
        "--save-png",
        "--no-save-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save x_pos + t_min trajectory plot per episode alongside the model (default True).",
    )
    args = parser.parse_args()

    mep = args.max_episode_steps if args.max_episode_steps is not None else config.training.max_episode_steps
    agent = ProximalPolicyOptimizationAgent.load(
        AntiPendulumEnv,
        model_path=args.model_path,
        env_kwargs={
            "crane": build_crane,
            "start_speed": args.start_speed,
            "randomize_start": args.randomize_start,
            "render_mode": args.render_mode,
            "reward_fac": config.reward,
            "rail_limit": config.training.rail_limit,
            "reward_limit": config.training.reward_limit,
            "continuous_actions": args.continuous_actions,
        },
        max_episode_steps=mep,
    )

    for episode in range(args.episodes):
        LOGGER.info("Episode %s/%s", episode + 1, args.episodes)
        png_path: str | None = None
        if args.save_png:
            stem = Path(args.model_path).stem
            png_path = str(
                Path(args.model_path).parent
                / f"{stem}_play_ss{args.start_speed:+.1f}_ep{episode + 1}.png"
            )
        agent.do_one_episode(seed=episode + 1, save_png=png_path)
        LOGGER.info("  start_speed: %.3f", agent.env.unwrapped.initial_speed)  # type: ignore[attr-defined]
        if png_path is not None:
            LOGGER.info("  PNG: %s", png_path)


if __name__ == "__main__":
    main()
