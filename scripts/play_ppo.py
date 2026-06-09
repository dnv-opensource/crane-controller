"""Run a trained PPO agent on the AntiPendulumEnv.

Examples
--------
.. code-block:: bash

    uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip
    uv run python scripts/play_ppo.py --model-path models/ppo.zip --render-mode plot --episodes 3
"""

import argparse
import logging

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
    args = parser.parse_args()

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
            "continuous_actions": args.continuous_actions,
        },
    )

    for episode in range(args.episodes):
        LOGGER.info("Episode %s/%s", episode + 1, args.episodes)
        agent.do_one_episode(seed=episode + 1)
        LOGGER.info("  start_speed: %.3f", agent.env.unwrapped.initial_speed)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
