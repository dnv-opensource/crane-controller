"""Inspect a trained Q-table for the AntiPendulumEnv.

Prints a per-pos/speed average summary by default. Use --obs to drill
into specific states; negative values act as wildcards (match any).

The observation tuple has 5 dimensions:
  [energy, pos, speed, distance, sector]

Example:
    uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json
    uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json --obs -1 0 0 -1 -1
    uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json --obs -1 1 1 -1 -1
"""

import argparse
import logging

import numpy as np

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

LOGGER = logging.getLogger(__name__)


def _build_dummy_env() -> AntiPendulumEnv:
    """Minimal env needed to satisfy QLearningAgent constructor (action_space.n)."""
    return AntiPendulumEnv(build_crane, discrete=QLearningAgent.DEFAULT_DISCRETE.copy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a trained Q-table for the crane anti-pendulum task.")
    _ = parser.add_argument("--model-path", type=str, required=True, help="Path to a trained Q-table JSON")
    _ = parser.add_argument(
        "--obs",
        type=int,
        nargs=5,
        metavar=("ENERGY", "POS", "SPEED", "DISTANCE", "SECTOR"),
        help="Filter Q-values for a specific observation (use -1 as wildcard)",
    )
    args = parser.parse_args()

    env = _build_dummy_env()
    agent = QLearningAgent(env, trained=(args.model_path, True))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    LOGGER.info("Q-table: %s states  (%s)", len(agent.q_values), args.model_path)
    LOGGER.info("")

    if args.obs:
        LOGGER.info("Q-values matching obs %s  (columns: state, q-values, best-action, mean, cv)", args.obs)
        LOGGER.info("%s", "-" * 72)
        agent.analyse_q(tuple(args.obs))
    else:
        LOGGER.info("Per-pos/speed average Q-values  (actions: left=0, coast=1, right=2)")
        LOGGER.info("%s", "-" * 52)
        for pos in (0, 1):
            for speed in (0, 1):
                res = {k: v for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed}
                avgs = [np.average([x[i] for x in res.values()]) for i in range(3)]
                LOGGER.info("  pos=%s  speed=%s  ->  %s", pos, speed, [f"{a:.4f}" for a in avgs])


if __name__ == "__main__":
    main()
