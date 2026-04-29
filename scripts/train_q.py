"""Train a Q-learning agent on the AntiPendulumEnv.

Examples
--------
.. code-block:: bash

    uv run python scripts/train_q.py
    uv run python scripts/train_q.py --episodes 50000 --v0 1.0 --save-path models/q_start.json
    uv run python scripts/train_q.py --trained models/q_AntiPendulumEnv.json
    uv run python scripts/train_q.py --intervals 10
    uv run python scripts/train_q.py --dry-run
"""

import argparse
import logging
from pathlib import Path

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and train a Q-learning agent."""
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--episodes", type=int, default=10_000, help="Total training episodes")
    _ = parser.add_argument("--v0", type=float, default=-1.0, help="Initial crane speed (negative = stop mode)")
    _ = parser.add_argument(
        "--reward-limit", type=float, default=-0.05, help="Per-episode reward termination threshold"
    )
    _ = parser.add_argument(
        "--save-path",
        type=str,
        default="models/q_AntiPendulumEnv.json",
        help="Where to save the trained Q-table",
    )
    _ = parser.add_argument(
        "--trained", type=str, default=None, help="Path to an existing Q-table JSON to continue from"
    )
    _ = parser.add_argument(
        "--intervals",
        type=int,
        default=0,
        help="Run interval training: N intervals of 10 episodes each (0 = disabled)",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 50 episodes with a reward plot and no model saved, for a quick visual sanity check.",
    )
    args = parser.parse_args()

    env = AntiPendulumEnv(
        build_crane,
        start_speed=args.v0,
        render_mode="plot" if args.dry_run else "none",
        reward_limit=args.reward_limit,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )

    if args.dry_run:
        agent = QLearningAgent(env, trained=None)
        agent.do_episodes(n_episodes=50, max_steps=1000)

    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        trained = (args.trained, True) if args.trained else (args.save_path, False)
        agent = QLearningAgent(env, trained=trained)
        agent.do_episodes(n_episodes=args.episodes, max_steps=5000)
        LOGGER.info(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
