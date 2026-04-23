"""Run a trained Q-learning agent on the AntiPendulumEnv.

Example:
    uv run python scripts/play_q.py --model-path models/q_AntiPendulumEnv.json
    uv run python scripts/play_q.py --model-path tests/anti-pendulum.json --render-mode plot --episodes 3
"""

import argparse
import logging

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Q-learning agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--model-path", type=str, required=True, help="Path to a trained Q-table JSON")
    _ = parser.add_argument(
        "--render-mode", type=str, default="plot", help="Render mode (plot, play-back, reward-tracking)"
    )
    _ = parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    _ = parser.add_argument("--v0", type=float, default=-1.0, help="Initial crane speed (negative = stop mode)")
    args = parser.parse_args()

    env = AntiPendulumEnv(
        build_crane,
        start_speed=args.v0,
        render_mode=args.render_mode,
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )
    agent = QLearningAgent(env, trained=(args.model_path, True))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for episode in range(args.episodes):
        LOGGER.info("Episode %s/%s", episode + 1, args.episodes)
        _ = env.reset(seed=episode + 1)
        agent.do_episodes(n_episodes=1)


if __name__ == "__main__":
    main()
