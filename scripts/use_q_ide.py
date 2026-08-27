"""Train a Q-learning agent on the AntiPendulumEnv. Variant of train_q.py, running directly in the IDE.

Examples:
--------
See end of the file, commented out code.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from crane_controller.crane_factory import build_crane
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv
from crane_controller.envs.simple_test_env import SimpleTestEnv
from crane_controller.q_agent import QLearningAgent, QLearningConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
MODELS = Path(__file__).parent.resolve().parent / "models"
USE_DISCRETE2 = 2
RELAX_LIMIT = 100


def analyse_trained(
    filename: str, episodes: int, *, r_limit: float|None = None, randomize_start: bool = False, show: bool = False
) -> str:
    """Perform the analysis for the report on one trained data set, providing a string on results.

    Args:
        filename: name of the json file to analyse. MODELS path is added automatically
        episodes: the number of episodes to run for the analysis
        r_limit: the reward_limit to use (independet of the limit used during training)
        randomize_start: whether to used randomized start speed of load
        show: whether to show a result plot

    Returns:
        a summary string (used in report latex table)
    """
    file = MODELS / filename
    assert file.exists(), f"File {file} not found"
    info, q_values = QLearningAgent.read_dumped(file)
    info["environment"]["render_mode"] = "none"
    info["q_agent"]["use_file"] = "r"  # keep file unchanged
    info["q_agent"]["auto_run"] = episodes
    if r_limit is not None:
        info["environment"]["reward_limit"] = r_limit  # standardized for analysis
    info["environment"]["randomize_start"] = randomize_start
    env = AntiPendulumEnv(build_crane, conf=AntiPendulumConfig(**info["environment"]))
    _agent = QLearningAgent(env, conf=QLearningConfig(**info["q_agent"]), q_values=q_values)
    relax: list[float] = [r for r in env.reward_stats["relaxation"] if abs(r) < RELAX_LIMIT]
    term_time: list[int] = []
    for t, s in zip(env.reward_stats["steps"], env.reward_stats["status"], strict=True):
        if s == 1:
            term_time.append(t)

    txt = f"{filename[15:-5]} & "
    txt += f"{info['environment']['reward_fac']['position']} & "
    txt += f"{info['environment']['discount']} & "
    txt += f"{sum(x == -1 for x in env.reward_stats['status']) / episodes: 2.0f} & "
    txt += f"{sum(x == 1 for x in env.reward_stats['status']) / episodes: 2.0f} & "
    txt += f"{np.average(term_time):2.2f} & "
    txt += f"{np.average(relax):2.2f} +/- {np.std(relax):2.2f} & "
    if show:
        _ = plt.plot(np.arange(len(relax)), relax, label="relaxation")
        _ = plt.legend()
        plt.show()
    return txt


def analyse_all(episodes: int = 1000, r_limit: float = -0.01) -> None:
    """Analyse all q_anti-pendulum*.json files in folder.

    Args:
        episodes: number of episodes to use in analysis
        r_limit: common r_limit to use, independent of training r_limit.
    """
    header = "ID & position & discount & trunc & term & avg.time & relaxation & "
    rows: list[str] = [header]
    for file in MODELS.glob("q_anti-pendulum*.json"):
        LOGGER.info(f"Analyse {file.name}")
        txt = analyse_trained(file.name, episodes, r_limit=r_limit, show=False)
        rows.append(txt)
    for r in rows:
        LOGGER.info(r)


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
    # ruff: disable[ERA001]  ## we intentionally work with commenting out lines here. Long lines allowed
    # ruff: disable[E501] ## allow long lines so that the whole command can be commented out
    run = QLearningAgent.auto_run  # alias for the auto_run on configuration function
    # run( start_speed, render_mode, file, use_file, episodes, steps, reward_fac, reward, s, seed, )
    ## Anti-pendulum training and results:
    # run(conf=MODELS/"q_anti-pendulum15.json")
    # run(conf=MODELS/"q_anti-pendulum3.json", _env={"render_mode": "plot"}, _agent={"use_file": "r", "auto_run": 10})
    # run(conf=MODELS/"q_anti-pendulum8.json", _env={"render_mode": "plot"}, _agent={"use_file": "r", "auto_run": 1})
    # run(conf=MODELS/"q_anti-pendulum8.json", _env={"render_mode": "plot", "randomize_start":True}, _agent={"use_file": "r", "auto_run": 10})

    # print(analyse_trained("q_anti-pendulum15.json", episodes=1000, show=True))
    print(analyse_trained("q_anti-pendulum2.json", episodes=1000, randomize_start=True, show=True))  # noqa: T201
    # analyse_all()
    ## Pendulum training and results:
    # conf0 = update_conf(conf1, {'start_speed':0.0,'file':MODELS / "q_pendulum.json",'reward_limit':1000.0})
    # run( update_conf( conf0, {'use_file':"r", 'episodes':10,'render_mode':'plot'}))
    # run(conf0)
    # simple_env(episodes=50000, render_mode="none", file=models/"q_simple.json", use="w", reward_limit=29.4, steps=200)
    # simple_env(episodes=10, render_mode="plot", file=models/"q_simple.json", use="r", reward_limit=29.7, steps=20)
    # ruff: enable[ERA001]
    # ruff: enable[E501]
