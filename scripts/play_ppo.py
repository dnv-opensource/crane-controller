"""Run a trained PPO agent on the AntiPendulumEnv.

Examples
--------
.. code-block:: bash

    uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip
    uv run python scripts/play_ppo.py --model-path models/ppo.zip --render-mode plot --episodes 3
    uv run python scripts/play_ppo.py --model-path models/ppo.zip --speed-sweep --render-mode none
"""

import argparse
import csv
import dataclasses
import logging
from pathlib import Path

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.experiment_config import load_training_sidecar
from crane_controller.ppo_agent import EpisodeResult, ProximalPolicyOptimizationAgent

LOGGER = logging.getLogger(__name__)

SWEEP_SPEEDS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def main() -> None:
    """Parse CLI arguments and run a trained PPO agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Pre-parse --model-path so the sidecar can seed argument defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    _ = pre_parser.add_argument("--model-path", type=str, required=True)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_training_sidecar(pre_args.model_path)

    parser = argparse.ArgumentParser(description="Run a trained PPO agent on the crane anti-pendulum task.")
    _ = parser.add_argument("--model-path", type=str, required=True, help="Path to a trained .zip model")
    _ = parser.add_argument("--render-mode", type=str, default="play-back", help="Render mode for playback")
    _ = parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run per speed")
    _ = parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=config.training.randomize_start,
        help="Randomise initial pendulum speed each episode (default from model sidecar).",
    )
    _ = parser.add_argument(
        "--start-speed",
        type=float,
        default=1.0,
        help="Initial pendulum speed for playback (default 1.0). Ignored when --speed-sweep is set.",
    )
    _ = parser.add_argument(
        "--continuous-actions",
        "--no-continuous-actions",
        action=argparse.BooleanOptionalAction,
        default=config.training.continuous_actions,
        help="Use Box(-1,1) action space (default from model sidecar).",
    )
    _ = parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "Override TimeLimit for play episodes (default: value from model sidecar). "
            "Pass 3000 when playing old pre-trained models that have no max_episode_steps "
            "in their sidecar (those default to 100 otherwise)."
        ),
    )
    _ = parser.add_argument(
        "--save-png",
        "--no-save-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 7-panel trajectory plot per episode alongside the model (default True).",
    )
    _ = parser.add_argument(
        "--save-csv",
        "--no-save-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-episode metrics to {stem}_play_results.csv alongside the model (default True).",
    )
    _ = parser.add_argument(
        "--speed-sweep",
        action="store_true",
        default=False,
        help=f"Run over speeds {SWEEP_SPEEDS} instead of --start-speed.",
    )
    args = parser.parse_args()

    mep = args.max_episode_steps if args.max_episode_steps is not None else config.training.max_episode_steps
    speeds = SWEEP_SPEEDS if args.speed_sweep else [args.start_speed]

    agent = ProximalPolicyOptimizationAgent.load(
        AntiPendulumEnv,
        model_path=args.model_path,
        env_kwargs={
            "crane": build_crane,
            "start_speed": speeds[0],
            "randomize_start": args.randomize_start,
            "render_mode": args.render_mode,
            "reward_fac": config.reward,
            "rail_limit": config.training.rail_limit,
            "reward_limit": config.training.reward_limit,
            "continuous_actions": args.continuous_actions,
        },
        max_episode_steps=mep,
    )

    stem = Path(args.model_path).stem
    all_results: list[EpisodeResult] = []

    for speed in speeds:
        agent.env.unwrapped.start_speed = speed  # type: ignore[attr-defined]
        for episode in range(args.episodes):
            LOGGER.info("Episode %s/%s  speed=%+.1f", episode + 1, args.episodes, speed)
            png_path: str | None = None
            if args.save_png:
                png_path = str(
                    Path(args.model_path).parent / f"{stem}_play_ss{speed:+.1f}_ep{episode + 1}.png"
                )
            result = agent.do_one_episode(seed=episode + 1, save_png=png_path)
            LOGGER.info(
                "  steps=%d  rew=%.2f  t_min=[%.2f→%.2f]  terminated=%s  truncated=%s",
                result.ep_steps,
                result.ep_reward,
                result.t_min_start,
                result.t_min_final,
                result.terminated,
                result.truncated,
            )
            if png_path is not None:
                LOGGER.info("  PNG: %s", png_path)
            all_results.append(result)

    if args.save_csv and all_results:
        csv_path = Path(args.model_path).parent / f"{stem}_play_results.csv"
        fieldnames = [f.name for f in dataclasses.fields(EpisodeResult)]
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataclasses.asdict(r) for r in all_results)
        LOGGER.info("Play CSV: %s", csv_path)


if __name__ == "__main__":
    main()
