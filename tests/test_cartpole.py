import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: F401

from crane_controller.envs.cart_pole import CartPoleEnv
from crane_controller.ppo_agent import ProximalPolicyOptimizationAgent

np.set_printoptions(formatter={"float_kind": "{:.4f}".format})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


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


def test_init(show: bool = False):
    """Test the initialization of the environment."""
    env = CartPoleEnv(seed=1)
    assert env.action_space.shape is not None and not len(env.action_space.shape)
    rnd_u = env.np_random.uniform(2, 8)
    rnd_r = env.np_random.random()
    assert rnd_u == 3.8709887120629127, f"Returns pseudo-random numbers when seed is given. Got {rnd_u} for seed 1"
    assert rnd_r == 0.42332644897257565, f"Returns pseudo-random numbers when seed is given. Got {rnd_r} for seed 1"
    obs, inf = env.reset(seed=1)
    expected = [
        0.0011821624357253313,
        0.0450463704764843,
        -0.035584039986133575,
        0.044864945113658905,
    ]
    assert np.allclose(obs, expected), f"Found {obs}"
    obs, reward, terminated, truncated, info = env.step(1)
    assert np.allclose(obs, (0.00208309, 0.24066003, -0.03468674, -0.2588293))
    assert reward == 1.0
    rewards = []
    cnt = 0
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(env.np_random.integers(0, 2))
        rewards.append(reward)
        cnt += 1
        if cnt > 100:
            break
    if show:
        show_figure(times=np.linspace(0, cnt + 1, cnt + 1), traces={"rewards": rewards})
    env.reset()


def test_training(agent: ProximalPolicyOptimizationAgent):
    logger.info(f"Agent {agent.env.__name__} initialized. Start training...")
    agent.do_training()
    logger.info("Training done")


def test_act(agent: ProximalPolicyOptimizationAgent):
    logger.info(f"Agent {agent.env} initialized. Start action...")
    agent.env.render_mode = "human"
    agent.do_one_episode()
    logger.info("Episode done")


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    agent = ProximalPolicyOptimizationAgent(
        CartPoleEnv,  # type: ignore[arg-type]  # do not know how to do that differently
        n_envs=0,
        env_kwargs={"seed": 1, "render_mode": "human"},
        trained=("ppo_CartPoleEnv", True),
    )
    test_init(show=True)
    # test_training(agent)
    # test_act(agent)
