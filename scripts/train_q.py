"""Train a Q-learning agent on the AntiPendulumEnv.

Example:
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

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on the crane anti-pendulum task.")
    parser.add_argument("--episodes", type=int, default=10_000, help="Total training episodes")
    parser.add_argument("--v0", type=float, default=-1.0, help="Initial crane speed (negative = stop mode)")
    parser.add_argument("--reward-limit", type=float, default=-0.05, help="Per-episode reward termination threshold")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/q_AntiPendulumEnv.json",
        help="Where to save the trained Q-table",
    )
    parser.add_argument("--trained", type=str, default=None, help="Path to an existing Q-table JSON to continue from")
    parser.add_argument(
        "--intervals",
        type=int,
        default=0,
        help="Run interval training: N intervals of 10 episodes each (0 = disabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 50 episodes with a reward plot and no model saved, for a quick visual sanity check.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

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

    elif args.intervals > 0:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent = QLearningAgent(env, trained=(args.save_path, False))
        for i in range(args.intervals):
            env.reset(seed=i + 1)
            agent.do_episodes(n_episodes=10)
            if i == 0:
                agent = QLearningAgent(env, trained=(args.save_path, True))
        LOGGER.info("Model saved to %s", args.save_path)

    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        trained = (args.trained, True) if args.trained else (args.save_path, False)
        agent = QLearningAgent(env, trained=trained)
        agent.do_episodes(n_episodes=args.episodes, max_steps=5000)
        LOGGER.info("Model saved to %s", args.save_path)


if __name__ == "__main__":
    main()
