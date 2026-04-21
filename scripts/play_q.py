"""Run a trained Q-learning agent on the AntiPendulumEnv.

Example:
    uv run python scripts/play_q.py --model-path models/q_AntiPendulumEnv.json
    uv run python scripts/play_q.py --model-path tests/anti-pendulum.json --render-mode plot --episodes 3
"""

import argparse

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
    parser = argparse.ArgumentParser(description="Run a trained Q-learning agent on the crane anti-pendulum task.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a trained Q-table JSON")
    parser.add_argument("--render-mode", type=str, default="plot", help="Render mode (plot, play-back, reward-tracking)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--v0", type=float, default=-1.0, help="Initial crane speed (negative = stop mode)")
    args = parser.parse_args()

    env = AntiPendulumEnv(
        _build_crane,
        start_speed=args.v0,
        render_mode=args.render_mode,
        discrete=_DISCRETE.copy(),
    )
    agent = QLearningAgent(env, trained=(args.model_path, True))

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        env.reset(seed=episode + 1)
        agent.do_episodes(n_episodes=1)


if __name__ == "__main__":
    main()
