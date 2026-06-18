"""Train a Q-learning agent on the AntiPendulumEnv. Variant of train_q.py, running directly in the IDE.

Examples:
--------
See end of the file, commented out code.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv
from crane_controller.envs.simple_test_env import SimpleTestEnv
from crane_controller.experiment_config import RewardConfig
from crane_controller.q_agent import QLearningAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
MODELS = Path(__file__).parent.resolve().parent / "models"
USE_DISCRETE2 = 2


@dataclass(kw_only=True, frozen=True, slots=True)
class Config:
    """Data for experiments performed in this module.

    Args:
        v0: start speed of load in x-direction. 0: Pendulum mode, >/< 0 same/random start at every episode
        randomize_start: Optionally randomize the start speed within +/- v0. Default: False
        render: render mode of environment
        file: Optional definition of model-save file
        use_file: How 'file' is used (if exists): 'r', 'w', 'rw'
        episodes: nnumber of episodes run in the training
        steps: number of steps per episodes (if not terminated or truncated)
        dt: step-size per time step
        r_fac: optional weight factors (RewardConfig) for reward
        r_limit: optional reward limit
        disc: discount rate of acceleration history to include in observation
        lr: optionally change the learning rate
        seed: optionally change the start seed

    """

    v0: float = 1.0
    randomize_start: bool = False
    render: str = "none"
    discretization: str = "energy"
    file: str | None = None
    use_file: str = "r"
    episodes: int = 10000
    steps: int = 1000
    dt: float = 1.0
    rc: RewardConfig | None = None
    r_limit: float | None = None
    discount: float = 0.8
    seed: int = 1
    strategy: str = "default"
    lr: float = 0.1
    eps: float = 1e-10
    if rc is None:
        rc = RewardConfig(energy=1.0, positional=1.0, crane_velocity=0.5)


def do_use(conf: Config | dict[str, Any] | None = None) -> None:
    """Perform training on the (Anti-)Pendulum environment using q-learning.

    Args:
        conf: Configuration data set. See Config class for all definitions.
    """
    _conf = Config() if conf is None else (Config(**conf) if isinstance(conf, dict) else conf)
    env = AntiPendulumEnv(
        build_crane,
        start_speed=_conf.v0,
        randomize_start=_conf.randomize_start,
        seed=_conf.seed,
        dt=_conf.dt,
        render_mode=_conf.render,
        discrete=_conf.discretization,
        reward_fac=_conf.rc,
        reward_limit=_conf.r_limit,
        discount=_conf.discount,
    )

    filename = _conf.file
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    agent = QLearningAgent(env, filename=filename, use_file=_conf.use_file, strategy=_conf.strategy)
    LOGGER.info(f"DISCRETE: {agent.env.discrete}")
    agent.do_episodes(n_episodes=_conf.episodes, max_steps=_conf.steps, show=0)
    if filename is not None and "w" in agent.use_file:
        LOGGER.info(f"Model saved to {filename}")


def simple_env(episodes: int, render: str, file: str, use: str, r_limit: float | None, steps: int) -> None:
    """Define a SimpleTest environment.

    Args:
        episodes: number of episodes
        render: render mode
        file: Optional definition of model-save file
        use: How 'file' is used (if exists): 'r', 'w', 'rw'
        r_limit: optional reward limit
        steps: number of steps per episodes (if not terminated or truncated)
    """
    env = SimpleTestEnv(
        reward_fac=(1.0, 1.0),
        reward_limit=r_limit,
        dt=1.0,
        render_mode=render,
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
    # do_use( v0, render, file, use_file, episodes, steps, rc, reward, s, seed, )
    ## Anti-pendulum training and results:
    conf1 = {
        "discretization": "phase",
        "v0": 2.0,
        "render": "data",
        "file": MODELS / "q_anti-pendulum_2.json",
        "use_file": "rw",
        "episodes": 3000,
        "r_limit": -0.1,
        "seed": 43,
    }
    # do_use(conf1)
    do_use(update_conf(conf1, {"use_file": "r", "episodes": 10, "render": "plot"}))
    ## Pendulum training and results:
    # conf0 = update_conf(conf1, {'v0':0.0,'file':MODELS / "q_pendulum.json",'r_limit':1000.0}) # start a pendulum
    # do_use( update_conf( conf0, {'use_file':"r", 'episodes':10,'render':'plot'}))
    # do_use(conf0)
    # simple_env(episodes=50000, render="none", file=models/"q_simple.json", use="w", r_limit=29.4, steps=200)
    # simple_env(episodes=10, render="plot", file=models/"q_simple.json", use="r", r_limit=29.7, steps=20)
    # ruff: enable[ERA001]
