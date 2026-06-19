"""Train a Q-learning agent on the AntiPendulumEnv. Variant of train_q.py, running directly in the IDE.

Examples:
--------
See end of the file, commented out code.
"""

import logging
from pathlib import Path
from typing import Any

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv
from crane_controller.envs.simple_test_env import SimpleTestEnv
from crane_controller.experiment_config import RewardConfig
from crane_controller.q_agent import QLearningAgent, QLearningConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
MODELS = Path(__file__).parent.resolve().parent / "models"
USE_DISCRETE2 = 2


def do_use(conf: dict[str, Any]) -> None:
    """Perform training on the (Anti-)Pendulum environment using q-learning.

    Args:
        conf: Configuration data set. See Config class for all definitions.
    """
    _e_conf = AntiPendulumConfig()  # default values
    e_conf = AntiPendulumConfig(
        acc=conf.get("acc", _e_conf.acc),
        start_speed=conf.get("start_speed", _e_conf.start_speed),
        randomize_start=conf.get("randomize_start", _e_conf.randomize_start),
        render_mode=conf.get("render_mode", _e_conf.render_mode),
        rail_limit=conf.get("rail_limit", _e_conf.rail_limit),
        seed=conf.get("seed", _e_conf.seed),
        reward_limit=conf.get("reward_limit", _e_conf.reward_limit),
        dt=conf.get("dt", _e_conf.dt),
        discrete=conf.get("discrete", _e_conf.discrete),
        reward_fac=conf.get("reward_fac", _e_conf.reward_fac),
        continuous_actions=conf.get("continuous_actions", _e_conf.continuous_actions),
        length=conf.get("length", _e_conf.length),
        q_factor=conf.get("q_factor", _e_conf.q_factor),
    )
    env = AntiPendulumEnv(build_crane, conf=e_conf)
    _a_conf = QLearningConfig()  # default values
    a_conf = QLearningConfig(
        learning_rate=conf.get("learning_rate", _a_conf.learning_rate),
        epsilon_decay=conf.get("epsilon_decay", _a_conf.epsilon_decay),
        final_epsilon=conf.get("final_epsilon", _a_conf.final_epsilon),
        discount_factor=conf.get("discount_factor", _a_conf.discount_factor),
    )
    filename = conf.get("file")
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    agent = QLearningAgent(
        env,
        conf=a_conf,
        filename=filename,
        use_file=conf.get("use_file", "w"),
        strategy=conf.get("strategy", "default"),
    )
    LOGGER.info(f"DISCRETE: {agent.env.discrete}")
    agent.do_episodes(n_episodes=conf.get("episodes", 10), max_steps=conf.get("steps", 1000), show=0)
    if filename is not None and "w" in agent.use_file:
        LOGGER.info(f"Model saved to {filename}")


def simple_env(episodes: int, render_mode: str, file: str, use: str, reward_limit: float | None, steps: int) -> None:
    """Define a SimpleTest environment.

    Args:
        episodes: number of episodes
        render_mode: render_mode mode
        file: Optional definition of model-save file
        use: How 'file' is used (if exists): 'r', 'w', 'rw'
        reward_limit: optional reward limit
        steps: number of steps per episodes (if not terminated or truncated)
    """
    env = SimpleTestEnv(
        reward_fac=(1.0, 1.0),
        reward_limit=reward_limit,
        dt=1.0,
        render_mode=render_mode,
    )
    agent = QLearningAgent(env, filename=file, use_file=use)
    agent.do_episodes(n_episodes=episodes, max_steps=steps)


def update_conf(conf: dict["str", Any], updates: dict["str", Any]) -> dict["str", Any]:
    """Update a dict and return it."""
    _conf = conf.copy()
    _conf.update(updates)
    return _conf


if __name__ == "__main__":
    # ruff: disable[ERA001]  ## we intentionally work with commenting out lines here
    # do_use( start_speed, render_mode, file, use_file, episodes, steps, reward_fac, reward, s, seed, )
    ## Anti-pendulum training and results:
    conf1 = {
        "discrete": "phase",
        "start_speed": 2.0,
        "randomize_start": False,
        "render_mode": "data",
        "file": MODELS / "q_anti-pendulum1.json",
        "use_file": "rw",
        "steps": 1000,
        "episodes": 50000,
        "reward_fac": RewardConfig(energy=1.0, positional=1.0, crane_velocity=0.5),
        "reward_limit": -0.001,
        "seed": 43,
        "q_factor": 500,
    }
    _conf1 = update_conf(conf1, {"use_file": "r", "episodes": 10, "render_mode": "plot"})
    # do_use(conf1)
    do_use(_conf1)
    # do_use(update_conf(conf1, {"use_file": "r", "episodes": 10, "render_mode": "plot"}))
    # conf2 = update_conf(conf1, {"file": MODELS / "q_anti-pendulum2.json", "randomize_start": True})
    # do_use(conf2)

    ## Pendulum training and results:
    # conf0 = update_conf(conf1, {'start_speed':0.0,'file':MODELS / "q_pendulum.json",'reward_limit':1000.0})
    # do_use( update_conf( conf0, {'use_file':"r", 'episodes':10,'render_mode':'plot'}))
    # do_use(conf0)
    # simple_env(episodes=50000, render_mode="none", file=models/"q_simple.json", use="w", reward_limit=29.4, steps=200)
    # simple_env(episodes=10, render_mode="plot", file=models/"q_simple.json", use="r", reward_limit=29.7, steps=20)
    # ruff: enable[ERA001]
