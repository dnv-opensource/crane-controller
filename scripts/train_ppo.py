"""Train a PPO agent on the AntiPendulumEnv.

Example:
    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --steps 500000 --n-envs 8 --save-path models/ppo.zip
"""

import argparse
from pathlib import Path

import numpy as np
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent


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
    parser = argparse.ArgumentParser(description="Train a PPO agent on the crane anti-pendulum task.")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--render-mode", type=str, default="none", help="Render mode during training")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_AntiPendulumEnv.zip",
        help="Where to save the trained model",
    )
    args = parser.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type]
        n_envs=args.n_envs,
        env_kwargs={
            "crane": _build_crane,
            "start_speed": 1.0,
            "render_mode": args.render_mode,
        },
        trained=(args.save_path, True),
    )
    agent.do_training(args.steps)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
