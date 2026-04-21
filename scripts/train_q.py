"""Train a Q-learning agent on the AntiPendulumEnv.

Example:
    uv run python scripts/train_q.py
    uv run python scripts/train_q.py --episodes 50000 --v0 1.0 --save-path models/q_start.json
    uv run python scripts/train_q.py --trained models/q_AntiPendulumEnv.json
    uv run python scripts/train_q.py --intervals 10
    uv run python scripts/train_q.py --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

_DISCRETE = {
    "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
    "pos": (0, 1),
    "speed": (0, 1),
    "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "sector": (0, 1),
}


def _build_crane(length: float = 10.0, mass: float = 1.0, q_factor: float = 50.0) -> Crane:
    crane = Crane()
    crane.add_boom(
        "pedestal",
        description="A simple pole with same length as the wire",
        mass=100.0,
        boom=(length, 0.0, 0.0),
    )
    crane.add_boom(
        "wire",
        description="The wire fixed to the pole. Flexible connection",
        mass=mass,
        mass_center=1.0,
        boom=(length, np.pi, 0.0),
        q_factor=q_factor,
    )
    crane.calc_statics_dynamics(None)
    return crane


def main():
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

    crane = _build_crane()
    env = AntiPendulumEnv(
        crane,
        start_speed=args.v0,
        render_mode="plot" if args.dry_run else "none",
        reward_limit=args.reward_limit,
        discrete=_DISCRETE.copy(),
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
        print(f"Model saved to {args.save_path}")

    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        trained = (args.trained, True) if args.trained else (args.save_path, False)
        agent = QLearningAgent(env, trained=trained)
        agent.do_episodes(n_episodes=args.episodes, max_steps=5000)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
