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

import numpy as np

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

_DISCRETE = {
    "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
    "pos": (0, 1),
    "speed": (0, 1),
    "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "sector": (0, 1),
}


def _build_crane():
    from py_crane.crane import Crane

    crane = Crane()
    crane.add_boom("pedestal", mass=100.0, boom=(10.0, 0.0, 0.0))
    crane.add_boom("wire", mass=1.0, mass_center=1.0, boom=(10.0, np.pi, 0.0), q_factor=50.0)
    crane.calc_statics_dynamics(None)
    return crane


def _build_dummy_env():
    """Minimal env needed to satisfy QLearningAgent constructor (action_space.n)."""
    return AntiPendulumEnv(_build_crane, discrete=_DISCRETE.copy())


def main():
    parser = argparse.ArgumentParser(description="Inspect a trained Q-table for the crane anti-pendulum task.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a trained Q-table JSON")
    parser.add_argument(
        "--obs",
        type=int,
        nargs=5,
        metavar=("ENERGY", "POS", "SPEED", "DISTANCE", "SECTOR"),
        help="Filter Q-values for a specific observation (use -1 as wildcard)",
    )
    args = parser.parse_args()

    env = _build_dummy_env()
    agent = QLearningAgent(env, trained=(args.model_path, True))

    print(f"Q-table: {len(agent.q_values)} states  ({args.model_path})")
    print()

    if args.obs:
        print(f"Q-values matching obs {args.obs}  (columns: state, q-values, best-action, mean, cv)")
        print("-" * 72)
        agent.analyse_q(tuple(args.obs))
    else:
        print("Per-pos/speed average Q-values  (actions: left=0, coast=1, right=2)")
        print("-" * 52)
        for pos in (0, 1):
            for speed in (0, 1):
                res = {k: v for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed}
                avgs = [np.average([x[i] for x in res.values()]) for i in range(3)]
                print(f"  pos={pos}  speed={speed}  ->  {[f'{a:.4f}' for a in avgs]}")


if __name__ == "__main__":
    main()
