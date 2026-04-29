"""Train a Q-learning agent on the AntiPendulumEnv. Variant of train_q.py, running directly in the IDE.

Examples
--------
See end of the file, commented out code.
"""

import logging
from pathlib import Path
from typing import Any

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.q_agent import QLearningAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def do_use(kwargs: dict[str, Any]) -> None:
    """Perform training on the (Anti-)Pendulum environment using q-learning.

    Args:
        dry_run (bool)=False: True: perform only a short run with plotting
        v0 (float)=1.0: start speed of load in x-direction. 0: Pendulum mode, >/< 0 same/random start at every episode
        render (str)='none': render mode of environment
        reward (float)=-0.1: reward limit at which episode is terminated
        file (str): Optional definition of model-save file
        use_trained (bool): Use pre-trained data?
        episodes (int)=10000: nnumber of episodes run in the training
        steps (int)=5000: number of steps per episodes (if not terminated or truncated)

    """
    if "dry-train" in kwargs:  # Check training setup (over-write some parameters)
        kwargs.update({"render": "plot", "file": None, "use_trained": False, "episodes": 10, "steps": 1000})
    elif "dry_do" in kwargs:  # Run a few episodes on trained data (file can be set by caller)
        kwargs.update({"render": "plot", "use_trained": True, "episodes": 10, "steps": 1000})
    env = AntiPendulumEnv(
        build_crane,
        seed=1,
        dt=0.1,
        start_speed=kwargs.get("v0", 1.0),
        render_mode=kwargs.get("render", "none"),
        reward_limit=kwargs.get("reward", 0.0),
        discrete=QLearningAgent.DEFAULT_DISCRETE.copy(),
    )

    filename = kwargs.get("file")
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    use_trained = kwargs.get("use_trained", False)
    agent = QLearningAgent(env, filename=filename, use_trained=use_trained)
    agent.do_episodes(n_episodes=kwargs.get("episodes", 100), max_steps=kwargs.get("steps", 5000))
    if filename is not None:
        LOGGER.info(f"Model saved to {filename}")


if __name__ == "__main__":
    models = Path(__file__).parent.resolve().parent / "models"
    args = {
        "v0": 1.0,
        "render": "none",
        "reward": 0.0,
        "file": models / "q_anti-pendulum.json",
        "use_trained": True,
        "episodes": 1000,
        "steps": 5000,
    }
    # args.update({'episodes':6000, 'use_trained':True}) # noqa: ERA001 ## do a mayor training adding to data
    args.update({"episodes": 10, "render": "plot"})
    # args.update({'dry-train':True,}) # noqa: ERA001  ## check the setup before a long training
    # args.update({'dry_do':True}) # noqa: ERA001
    do_use(args)
