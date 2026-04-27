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
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and run a trained PPO agent."""
    parser = argparse.ArgumentParser(description="Run a trained PPO agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--model-path", type=str, required=True, help="Path to a trained .zip model")
    _ = parser.add_argument("--render-mode", type=str, default="play-back", help="Render mode for playback")
    _ = parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    agent = ProximalPolicyOptimizationAgent.load(
        AntiPendulumEnv,
        model_path=args.model_path,
        env_kwargs={
            "crane": build_crane,
            "start_speed": 1.0,
            "render_mode": args.render_mode,
        },
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for episode in range(args.episodes):
        LOGGER.info("Episode %s/%s", episode + 1, args.episodes)
        agent.do_one_episode(seed=episode + 1)


if __name__ == "__main__":
    main()
