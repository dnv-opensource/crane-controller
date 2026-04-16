import logging
from collections import defaultdict
from typing import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: F401

# from component_model.utils.controls import Controls
from py_crane.crane import Crane

from crane_controller.algorithm import AlgorithmAgent
from crane_controller.envs.controlled_crane_pendulum import AntiPendulumEnv

# from py_crane.animation import AnimateCrane
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent
from crane_controller.q_agent import QLearningAgent

np.set_printoptions(formatter={"float_kind": "{:.4f}".format})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_figure(
    times: np.ndarray | list[float],
    traces: dict[str, list[float]] | dict[str, np.ndarray],
    selection: dict[str, int] | None = None,
    title: str = "",
):
    """Plot selected traces."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None:  # all in first subplot
            _ = ax1.plot(times, trace, label=label)
        else:
            if label in selection:
                if selection[label] == 1:
                    _ = ax1.plot(times, trace, label=label)
                elif selection[label] == 2:
                    _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    plt.title(title)
    plt.show()


@pytest.fixture
def crane(scope: str = "session", autouse: bool = True):
    return _crane()


def _crane(length: float = 10.0, mass: float = 1.0, q_factor: float = 50.0):
    """Very simple mobile crane - actually only a pole and a wire for testing of anti-pendulum control.
    The crane has a pedestal and a wire of equal length.
    The size and weight of the various parts can be configured.

    Args:
        length (float) = 1.0 : height (fixed) of the pedestal and the wire
        mass (float) = 1.0 : mass of the load
        q_factor (float) = 50: The Q-factor (the pendulum damping)
    """

    crane = Crane()
    _ = crane.add_boom(
        "pedestal",
        description="A simple pole with same length as the wire",
        mass=100.0,
        boom=(length, 0.0, 0.0),
    )
    _ = crane.add_boom(
        "wire",
        description="The wire fixed to the pole. Flexible connection",
        mass=mass,
        mass_center=1.0,
        boom=(length, np.pi, 0.0),
        q_factor=q_factor,
    )
    crane.calc_statics_dynamics(None)  # make sure that _comSub is calculated for all booms
    return crane


def test_environment(crane: Callable, v0: float = 1.0, render_mode="plot", reward_limit=0.0):
    env = AntiPendulumEnv(
        crane,
        start_speed=v0,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "pos": (0, 1),
            "speed": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    assert env.action_space.n == 3 # type: ignore[attr-defined]
    assert env.action_space.start == 0 # type: ignore[attr-defined]
    assert env.action_space.dtype == np.int64
    assert isinstance(env.action_space.seed(), int)
    assert len(env.observation_space.nvec) == 5 # type: ignore[attr-defined]
    assert np.allclose(env.observation_space.nvec, [7, 2, 2, 6, 2]) # type: ignore[attr-defined]
    assert np.allclose(env.observation_space.start, [0, 0, 0, 0, 0]) # type: ignore[attr-defined]
    assert env.observation_space.dtype == np.int64
    assert isinstance(env.observation_space.seed(), int)
    q_values = defaultdict(lambda: np.array([env.low_reward()] * env.action_space.n))  # type: ignore  ## n!
    obs1 = np.array((0, 1, 1, 3, 0), int)
    obs2 = np.array((4, 0, 0, 1, 1), int)
    q_values[obs1.tobytes()]
    q_values[obs2.tobytes()]
    assert np.allclose(q_values[obs1.tobytes()], [-98.1000, -98.1000, -98.1000])
    assert q_values[obs2.tobytes()][2] == -98.1


def movement(crane: Crane, dt: float = 0.01, t_end: float = 10.0) -> Generator[tuple[float, Crane], None, None]:
    """Create movement of the crane through definition and usage of Controls.
    Generator function. Returns a `Generator` which steps through time
    and sequentially yields updated frames in the form of tuple (time, crane) objects.
    time is defined global as a simple way to draw the current time together with the title.
    """
    # initial definition of controls and start values
    f, p, w = list(crane.booms())

    # From time 0 we set three goals
    acc = -0.1
    crane.velocity = np.array((acc, 0.0, 0.0), float)
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
        if abs(time - int(time)) < 1e-6:  # switch acceleration
            acc = -acc
            crane.d_velocity[0] = acc
        crane.do_step(time, dt)
        yield (time + dt, crane)


# def test_pendulum_crane(crane: Crane, show: bool = False):
#     if not show:  # nothing to do
#         return
#     anim = AnimateCrane(
#         crane=crane,
#         movement=AntiPendulumEnv.movement,
#         dt=0.1,
#         t_end=10.0,
#         axes_lim=((-2, 2), (-2, 2), (0, 2)),
#         title="Pendulum-Crane animation",
#     )
#     anim.do_animation()


def test_init(crane: Crane, show: bool = False):
    """Test the initialization of the environment."""
    env = AntiPendulumEnv(crane, seed=1, start_speed=1.0, render_mode="play-back" if show else "data")
    rnd_u = env.np_random.uniform(2, 8)
    rnd_r = env.np_random.random()
    assert rnd_u == 5.07092974820154, f"Returns pseudo-random numbers when seed is given. Got {rnd_u} for seed 1"
    assert rnd_r == 0.9504636963259353, f"Returns pseudo-random numbers when seed is given. Got {rnd_r} for seed 1"
    obs, inf = env.reset(seed=1)
    assert np.allclose(obs, [0.0, 0.0, np.pi, 0.017453292519943295]), f"Found {obs[3]}"
    assert inf["steps"] == 0
    assert abs(inf["reward"] + 0.5 * 0.017453292519943295**2) < 1e-9, f"Found initial reward {inf['reward']}"
    obs, reward, terminated, truncated, info = env.step(-1)
    assert obs[0] == -0.1
    assert obs[1] == -0.1
    assert not terminated
    assert not truncated
    rewards = []
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.np_random.integers(-1, 2))
        rewards.append(reward)
    if show:
        show_figure(times=np.linspace(0, 100, 100), traces={"rewards": rewards})
    env.reset()


def test_monitor(crane: Crane):
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type] ## should be correct
        n_envs=1,
        env_kwargs={
            "crane": crane,
            "seed": 2,
            "start_speed": 1.0,
            "render_mode": "reward-tracking",
        },
    )
    agent.do_training(1000)


def test_algorithm_strategies(
    crane: Callable,
    render_mode: str = "plot",
    reward_limit: float = 1000.0,
    start_speed: float = 0.0,
):
    env = AntiPendulumEnv(
        crane,
        start_speed=start_speed,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "pos": (0, 1),
            "speed": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = AlgorithmAgent(env)
    agent.do_strategies()


def test_algorithm(crane: Callable, render_mode: str = "plot"):
    env = AntiPendulumEnv(
        crane,
        start_speed=0.0,
        render_mode=render_mode,
        reward_limit=1000.0,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "pos": (0, 1),
            "speed": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = AlgorithmAgent(env)
    agent.strategy = (0, 2, 0, 2)
    logger.info("Best strategy (0,2,0,2) in start mode")
    agent.do_episodes(1, max_steps=5000)
    agent.strategy = (1, 1, 1, 1)
    logger.info("Do-nothing strategy (1,1,1,1) in start mode")
    agent.do_episodes(1, max_steps=5000)
    env.start_speed = 1.0
    agent.strategy = (2, 1, 1, 0)
    logger.info("Best strategy (2,1,1,0) in stop mode")
    agent.do_episodes(1, max_steps=5000)
    agent.strategy = (1, 1, 1, 1)
    logger.info("Do-nothing strategy (1,1,1,1) in stop mode")
    agent.do_episodes(1, max_steps=5000)


def test_interval_training_q(crane: Crane, intervals=10, render_mode: str = "none", reward_limit=0.01):
    env = AntiPendulumEnv(
        crane,
        start_speed=-1.0,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "direction": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = QLearningAgent(env, trained=("interval.json", False))
    for i in range(intervals):
        logger.info(f"Start interval {i}...")
        env.reset(seed=i + 1)
        agent.do_episodes(n_episodes=10)
        if i == 0:
            agent = QLearningAgent(env, trained=("interval.json", True))  # use pre-trained from now on


def test_training_q(
    crane: Callable,
    episodes: int = 10000,
    render_mode: str = "none",
    reward_limit=-0.05,
    trained=None,
    v0: float = -1.0,
    max_steps: int = 1000,
    show: int = 0,
):
    env = AntiPendulumEnv(
        crane,
        start_speed=v0,
        render_mode=render_mode,
        reward_limit=reward_limit,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "pos": (0, 1),
            "speed": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = QLearningAgent(env, trained=trained)

    logger.info(f"Agent {agent.env} initialized. Start training...")
    agent.do_episodes(n_episodes=episodes, max_steps=max_steps, show=show)
    logger.info(f"Training done. Resets:{agent.env.nresets}, Successes:{agent.env.nsuccess}")  # type: ignore
    # agent.analyse_training()


def test_q_analyse(crane, trained: tuple[str, bool] = ("anti-pendulum.json", False)):
    env = AntiPendulumEnv(
        crane,
        discrete={
            "angles": (0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 90.0),
            "pos": (0, 1),
            "speed": (0, 1),
            "distance": (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
            "sector": (0, 1),
        },
    )
    agent = QLearningAgent(env, trained=trained)
    for k, v in agent.q_values.items():
        assert len(k) == 5, len(v) == 3
    for pos in (0, 1):
        for speed in (0, 1):
            res = dict((k, v) for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed)
            print(f"pos:{pos}, speed:{speed}")
            acc = []
            for i in range(3):
                col = [x[i] for x in res.values()]
                acc.append(np.average(col))
            print(acc)


def test_training_ppo(
    n_envs: int = 4,
    nsteps: int = 100000,
    render_mode: str = "data",
    trained: tuple[str, bool] | None = None,
):
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type] ## should be correct
        n_envs=n_envs,
        env_kwargs={
            "crane": crane,
            "start_speed": -1.0,
            "size": 20,
            "render_mode": render_mode,
        },
        trained=trained,
    )
    logger.info(f"Agent {agent.env} initialized. Start training...")
    agent.do_training(total_timesteps=nsteps, progress_bar=False)
    logger.info(f"Training done. Resets:{agent.env.nresets}, Successes:{agent.env.nsuccess}")


def test_act(
    render_mode: str = "play_back",
    trained: tuple[str, bool] = ("ppo_AntiPendulumEnv.zip", False),
    episodes: int = 1,
):
    agent = ProximalPolicyOptimizationAgent(
        AntiPendulumEnv,  # type: ignore[arg-type] ## should be correct
        n_envs=-1,
        env_kwargs={
            "crane": crane,
            "seed": 2,
            "start_speed": 1.0,
            "render_mode": render_mode,
        },
        trained=trained,
    )
    logger.info(f"Agent {type(agent.env).__name__} initialized. Start action...")
    for e in range(episodes):
        agent.do_one_episode(seed=e + 1)
        agent.env.reset()
    logger.info("Action done")


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_environment(_crane)
    # test_training_q(_crane,render_mode='reward-tracking', max_steps=1000, reward_limit=1.0, episodes=10, trained=("pendulum.json",False), v0=0.0)
    # test_q_analyse(_crane, trained=("pendulum.json",True))
    test_training_q(
        _crane,
        render_mode="plot",
        max_steps=1000,
        reward_limit=1.0,
        episodes=10,
        trained=("pendulum.json", True),
        v0=0.0,
    )

    # test_algorithm_strategies(_crane, render_mode="none", start_speed=0.0) # all combinations in start mode
    # test_algorithm_strategies(_crane, render_mode="none", start_speed=1.0) # all combinations in stop mode
    # test_algorithm(_crane, render_mode='plot') # test algorithmic with a few strategies
    # test_algorithm(_crane, render_mode='reward-tracking', start_speed=1.0, combination=(2,1,1,0)) # test in stop mode
    # test_interval_training_q(crane,render_mode='none', reward_limit=0.01, intervals=10)
    # test_training_q(_crane,render_mode='none', reward_limit=1000, episodes=5000, trained=("pendulum.json",True), v0=0.0)
    # test_q_analyse(_crane, trained=("pendulum.json",True))
    # test_training_q(_crane,render_mode="plot",reward_limit=1000,episodes=10,trained=("pendulum.json", False),v0=0.0)
    # test_training_q(_crane,render_mode='none', reward_limit=-0.001, episodes=10000, trained=("anti-pendulum.json",False))
    # test_q_analyse(_crane, trained=("anti-pendulum.json",True))
    # test_training_q(_crane,render_mode='plot', reward_limit=-0.0001, episodes=10, trained=("anti-pendulum.json",True))
    ## test_pendulum_crane(crane, show=True)
    # test_init(crane, show=True)
    # test_training_ppo(n_envs=1, nsteps=20000, render_mode='plot', trained=("antipendulum.zip",False))
    # test_training_ppo(n_envs=1, nsteps=500000, render_mode='data', trained=("antipendulum.zip",True))
    # test_training_ppo(n_envs=4, nsteps=100000, render_mode='data')
    # test_act(render_mode="reward-tracking", trained=("antipendulum.zip",False))
    # test_act(render_mode="plot", trained=("antipendulum.zip",False), episodes=1)
    # test_act(render_mode="plot", trained=("ppo_AntiPendulumEnv.zip",False), episodes=1)
    # test_monitor(crane)
