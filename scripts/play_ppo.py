"""Run a trained PPO agent on the AntiPendulumEnv.

Example:
    uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip
    uv run python scripts/play_ppo.py --model-path models/ppo.zip --render-mode plot --episodes 3
"""

import argparse

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
    parser = argparse.ArgumentParser(description="Run a trained PPO agent on the crane anti-pendulum task.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a trained .zip model")
    parser.add_argument("--render-mode", type=str, default="play-back", help="Render mode for playback")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type]
        n_envs=0,  # load-from-file mode
        env_kwargs={
            "crane": _build_crane,
            "start_speed": 1.0,
            "render_mode": args.render_mode,
        },
        trained=(args.model_path, True),
    )

    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        agent.do_one_episode(seed=episode + 1)


if __name__ == "__main__":
    main()
