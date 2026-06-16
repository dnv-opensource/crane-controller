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
from crane_controller.envs.simple_test_env import SimpleTestEnv
from crane_controller.q_agent import QLearningAgent
from crane_controller.experiment_config import RewardConfig


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
MODELS = Path(__file__).parent.resolve().parent / "models"



def do_use( v0 : float = 1.0,
            render : str = 'none',
            file : str|None = None,
            use_file : str = "r",
            episodes : int = 10000,
            steps : int = 5000,
            rc : RewardConfig = None,
            reward : float|None = None,
            disc : float = 0.8,
            lr : float = 0.1,
            seed : int = 1,
            s : int = 0
        ) -> None:
    """Perform training on the (Anti-)Pendulum environment using q-learning.

    Args:
        v0 (float)=1.0: start speed of load in x-direction. 0: Pendulum mode, >/< 0 same/random start at every episode
        render (str)='none': render mode of environment
        file (str): Optional definition of model-save file
        use_file (str): How 'file' is used (if exists): 'r', 'w', 'rw'
        episodes (int)=10000: nnumber of episodes run in the training
        steps (int)=5000: number of steps per episodes (if not terminated or truncated)
        fac (tuple[float,...])=(0.01,0.01),
        reward (float): optional reward limit
        disc (float) = 0.8: discount rate of acceleration history to include in observation
        lr (float) = 0.1: optionally change the learning rate
        seed (int) = 1: optionally change the start seed
    """
    env = AntiPendulumEnv(
        build_crane,
        seed=seed,
        dt=1.0,
        start_speed=v0,
        render_mode=render,
        discrete=AntiPendulumEnv.DEFAULT_DISCRETE if s!=2 else AntiPendulumEnv.DISCRETE2,
        reward_fac = rc,
        reward_limit = reward,
        discount = disc,
    )

    filename = file
    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    agent = QLearningAgent(env,
                           filename=filename,
                           use_file=use_file,
                           learning_rate=lr,
                           strategy = s)
    print("DISCRETE", agent.env.discrete)
    agent.do_episodes(n_episodes=episodes, max_steps=steps, show=0)
    if filename is not None and "w" in agent.use_file:
        LOGGER.info(f"Model saved to {filename}")
    #print("REVISED", agent.q_revised)

def simple_env( episodes:int, render:str, file:str, use:str, r_limit:float|None, steps:int):
    env = SimpleTestEnv(
        acc = 1.0,
        pos_range = (-100,100),
        speed_range = (-10,10),
        reward_fac = (1.0, 1.0),
        reward_limit = r_limit,
        dt = 1.0,
        pos0 = 0.0,
        speed0 = -5.0,
        pos1 = 10.0,
        speed1 = 0.0,
        render_mode = render
    )
    agent = QLearningAgent(env, filename=file, use_file=use)
    agent.do_episodes(n_episodes=episodes, max_steps=steps)


if __name__ == "__main__":

    # ruff: disable[ERA001]  ## we intentionally work with commenting out lines here
    # do_use( v0, render, file, use_file, episodes, steps, rc, reward, s, seed, )
    ## Anti-pendulum training and results:
    rc = RewardConfig(energy=1.0,positional=1.0,crane_velocity=0.5)
    do_use( 2, 'data', MODELS / "q_anti-pendulum_2.json", 'rw', 30000, 1000, rc, reward=-0.1, s=2, seed=43)
    #do_use( 2, 'plot', MODELS / "q_anti-pendulum_2.json", 'r', 10, 1000, rc, reward=-0.001, s=2)
    ## Pendulum training and results:
    # args = _args(pend, {'episodes':1000}) # pendulum training
    #args = _args( pend, {"episodes": 10, "render": "plot", "use_file":'r'}) # show start pendulum results
    # args = args.update(_args(anti, {'dry-train':True,})) # check the setup before a long training
    #simple_env(episodes=50000, render="none", file=models/"q_simple.json", use="w", r_limit=29.4, steps=200)
    #simple_env(episodes=10, render="plot", file=models/"q_simple.json", use="r", r_limit=29.7, steps=20)
    # ruff: enable[ERA001]
    
