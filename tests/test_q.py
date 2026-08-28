import datetime as dt
import itertools
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from py_crane.crane import Crane

from crane_controller.envs.controlled_crane_pendulum import AntiPendulumConfig, AntiPendulumEnv, _level
from crane_controller.experiment_config import RewardConfig
from crane_controller.q_agent import QLearningAgent, QLearningConfig

MODELS = Path(__file__).parent.resolve().parent / "models"

LOGGER = logging.getLogger(__name__)


def config_env_agent(
    crane: Callable[..., Crane],
    env_conf: dict[str, Any] | None = None,
    agent_conf: dict[str, Any] | None = None,
    file: Path | None = None,
    use_file: str = "r",
):
    """Make and configure the crane, the environment and the agent for various tests."""
    if env_conf is None:
        env_conf = {}
    if agent_conf is None:
        agent_conf = {}
    e_conf = AntiPendulumConfig(
        start_speed=env_conf.get("start_speed", 0.0),
        render_mode=env_conf.get("render_mode", "none"),
        reward_limit=env_conf.get("reward_limit", 0.0),
        discrete=env_conf.get("discrete", "phase"),
        q_factor=env_conf.get("q_factor", 10000.0),
    )

    env = AntiPendulumEnv(crane, e_conf)
    env.reset(options={"init": True})
    a_conf = QLearningConfig(discount_factor=agent_conf.get("discount_factor", 0.95))
    agent = QLearningAgent(env, a_conf, filename=file, use_file=use_file)
    return (env, agent)


def test_levels(crane: Callable[..., Crane]) -> None:
    def check(val: float, expected: int) -> None:
        assert _level(val, env.discrete["energy"]) == expected, f"Level {val} =? {_level(val, env.discrete['energy'])}"

    env = AntiPendulumEnv(crane, conf=AntiPendulumConfig(discrete="energy"))
    assert list(env.discrete.keys()) == ["energy", "distance", "pos", "speed", "c-pos", "c-speed", "avg-acc"], (
        f"Expected the 'energy' discretization. found {list(env.discrete.keys())}"
    )
    check(0, 0)
    check(-1e-10, -1)
    check(1e-10, 0)
    check(0.014, 0)
    check(0.015, 1)
    check(0.3, 1)
    check(0.4, 2)
    check(1.4, 2)
    check(1.5, 3)
    check(5.9, 3)
    check(6.0, 4)
    check(13.1, 4)
    check(13.2, 5)
    check(98, 5)
    check(99, -1)  # marks 'outside range' level
    check(float("inf"), -1)


def test_intervals(crane: Callable[..., Crane]):
    """Test that learning / saving / resuming learning works:"""
    save_path = Path.cwd() / "q_interval_training.json"
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="none",
            reward_limit=-0.05,
            discrete="energy",
            continuous_actions=False,
        ),
    )

    agent = QLearningAgent(
        env,
        conf=QLearningConfig(
            filename=str(save_path),
            use_file="w",
            learning_rate=0.1,
            epsilon_decay=1e-4,
            final_epsilon=0.1,
            discount_factor=0.95,
            auto_run=10,
        ),
    )
    epsilon0 = agent.conf.epsilon
    for i in range(10):
        assert agent.epsilon_decay == agent.conf.epsilon_decay, "Keeps unchanged"
        assert np.isclose(agent.epsilon, epsilon0 - 10 * (i + 1) * agent.conf.epsilon_decay), (
            f"{i}. Found {agent.epsilon} != {epsilon0 - 10 * (i + 1) * agent.conf.epsilon_decay}"
        )
        agent = QLearningAgent.auto_run(Path(save_path))
    LOGGER.info(f"Model saved to {save_path}")


def test_smoke(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="plot" if show else "none",
            reward_limit=-0.05,
            discrete="energy",
            continuous_actions=False,
        ),
    )
    agent = QLearningAgent(env, filename=None)
    agent.do_episodes(n_episodes=5, max_steps=200)


def test_q_analyse(crane: Callable[..., Crane], *, show: bool) -> None:
    models = Path(__file__).parent.resolve().parent / "models"
    assert (models / "q_trained.json").exists(), "Expect a file 'q_trained.json' in the models directory. Not found"
    _ = shutil.copy2(models / "q_trained.json", ".")  # copy to working_directory
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            discrete="energy",
            continuous_actions=False,
        ),
    )
    assert Path("q_trained.json").exists(), "File 'q_trained.json' not found"
    agent = QLearningAgent(env, filename=Path("q_trained.json"), use_file="r")
    _conf, agent.q_values = agent.read_dumped(Path("q_trained.json"))
    for k, v in agent.q_values.items():
        assert len(k) == 5, len(v) == 3
    for pos in (0, 1):
        for speed in (0, 1):
            res = {k: v for k, v in agent.q_values.items() if k[1] == pos and k[2] == speed}
            LOGGER.info(f"pos:{pos}, speed:{speed}")
            acc: list[np.floating] = []
            for i in range(3):
                col = [x[i] for x in res.values()]
                acc.append(np.average(col))
            LOGGER.info(f"averages: {acc}")


@pytest.mark.parametrize("discretization", ["energy", "phase"])
def test_discretization(crane: Callable[..., Crane], *, show: bool, discretization: str) -> None:
    """Test the discretization with respect to yielding unique rewards."""
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=2.0,
            render_mode="none",
            reward_limit=0.0,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.01}),
            discrete=discretization,
        ),
    )
    env.reset(options={"init": True})
    if discretization == "phase":  # not yet implemented
        return
    _agent = QLearningAgent(env)
    for e in range(len(env.discrete["energy"]) - 1):
        for s in range(len(env.discrete["speed"]) - 1):
            for c_p in range(len(env.discrete["c-pos"]) - 1):
                for c_s in range(len(env.discrete["c-speed"]) - 1):
                    action_sum = [0] * 3
                    for angle, speed, c_pos, c_speed in itertools.product(
                        (env.discrete["energy"][e], env.discrete["energy"][e + 1]),
                        (env.discrete["speed"][s], env.discrete["speed"][s + 1]),
                        (env.discrete["c-pos"][c_p], env.discrete["c-pos"][c_p + 1]),
                        (env.discrete["c-speed"][c_s], env.discrete["c-speed"][c_s + 1]),
                    ):
                        reward_max = float("-inf")
                        for action in range(3):
                            env.set_state(c_pos, c_speed, angle, float(speed))
                            _obs, reward, _term, _trunc, _ = env.step(action)
                            if reward > reward_max:
                                action_max = action
                                reward_max = reward
                        action_sum[action_max] += 1
                    if (
                        max(action_sum) != 16
                        and action_sum[0] > 0
                        and action_sum[2] > 0
                        and action_sum[0] == action_sum[2]
                    ):
                        LOGGER.info(f"angle:{e}, speed:{s}, c_pos:{c_p}, c_speed:{c_s}: {action_sum}")


def test_state(crane: Callable[..., Crane], *, show: bool) -> None:
    """Set state and calculate reward."""

    env, agent = config_env_agent(crane)

    env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.0)
    for _i in range(10):
        assert np.allclose(env.get_state(), (0, 0, 0, 0)), f"Found {env.get_state()}"
        env.step(1)  # check that nothing moves

    env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.1)
    w0 = np.sqrt(9.81 / 10)
    h_max = 0.5 / 9.81 * 0.1**2
    x_max = np.sqrt(2 * 10 * h_max)
    for t in range(50):
        env.step(1)
        state = env.get_state(as_x=True)
        assert np.allclose(state[:2], (0, 0))
        assert abs(state[2] - x_max * np.sin(w0 * (t + 1))) < 1e-3, (
            f"@{t + 1}: {state[2]} != {x_max * np.sin(w0 * (t + 1))}"
        )
        assert abs(state[3] - x_max * w0 * np.cos(w0 * (t + 1))) < 1e-3, (
            f"@{t + 1}: {state[3]} != {x_max * w0 * np.cos(w0 * (t + 1))}"
        )

    env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.0)
    assert np.allclose(env.get_state(), (0.0, 0.0, 0.0, 0.0)), f"Found {env.get_state()}"
    env.step(0)  # one negative acceleration step
    assert np.allclose(env.get_state(), (-0.1, -0.1, 0.0460491970, 0.08442572592699)), f"Found {env.get_state()}"

    env.set_state(pos=1.0, speed=2.0, direction=0.0, w_speed=0.0)
    assert np.allclose(env.get_state(), (1.0, 2.0, 0, 0)), f"Found {env.get_state()}"
    env.step(1)
    state = env.get_state()
    assert np.allclose(state, (3.0, 2.0, -0.46014216040346284, -0.8435711448494038)), f"Found {state}"

    env.set_state(0.0, 0.0, np.radians(10), -2.0)
    assert np.allclose(env.get_state(as_x=False), (0.0, 0.0, np.radians(10), -2)), f"Found {env.get_state(as_x=False)}"
    env.step(1)
    assert np.allclose(env.get_state()[:2], (0.0, 0.0)), f"Found {env.get_state()}"
    # env.set_state(pos=18.0, speed=0.0, direction=0.0, w_speed=0.0)
    # LOGGER.info( env.step(1))
    env.set_state(pos=2.0, speed=0.0, direction=0.0, w_speed=0.0)
    res = env.step(1)  # neutral step
    actions = (
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        1,
    )
    reward = float("-inf")
    reward_sum = 0.0
    for s in range(len(actions)):
        obs0, _reward0 = res[:2]
        res = env.step(actions[s])
        if res[1] <= reward:
            LOGGER.info(f"step:{s}, actions:{actions[:s]}, obs:{res[0]}, reward:{float(res[1])}")
            break
        assert isinstance(obs0, tuple)
        assert isinstance(res[0], tuple)
        agent.update_q(obs0, actions[s], res[1], terminated=False, s1=res[0], r0=reward)
        reward = float(res[1])
        reward_sum += reward
    LOGGER.info(f"reward:{reward}, avg:{reward_sum / s}")
    LOGGER.info(f"pos:{env.crane.position}, speed:{env.crane.velocity}, dir:{env.wire.direction}, v_w:{env.wire.cm_v}")
    for k, v in agent.q_values.items():
        LOGGER.info(f"key:{k}, value:{v}")
    # env.set_state(pos=18.0, speed=0.0, direction=0.0, w_speed=0.0)
    # LOGGER.info( env.step(2))


def test_state_sequence1(crane: Callable[..., Crane], *, show: bool) -> None:
    """Go through an algorithmic action sequence and use that to investigate the q_agent."""

    steps: int = 100
    env, agent = config_env_agent(
        crane,
        env_conf={"render_mode": "plot", "start_speed": 2.0},
        agent_conf={"discount_factor": 0.95, "strategy": "default"},
        file=Path(__file__).parent.resolve().parent / "models/q_sequence.json",
        use_file="w",
    )
    seq: list[int] = []
    r0 = env.reward
    obs0 = env.obs
    assert isinstance(obs0, tuple)
    LOGGER.info(f"Obs0:{obs0}, r0:{r0}, state:{env.get_state()}, Q:{agent.q_values[obs0]}")
    for i in range(steps):
        c_x, c_v, x, v = env.get_state()
        phase = np.arctan2(v, x)
        a_x = -2 * (c_x + c_v * env.conf.dt) / env.conf.dt**2  # acceleration to bring crane to origin in dt
        a_v = -c_v / env.conf.dt  # acceleration to stop crane
        a_sign = (phase if abs(phase) > 0.5 else 0) + 0.3 * a_x / env.conf.acc + 1.0 * a_v / env.conf.acc
        #        a_sign = (phase if abs(phase)>1.0 else 0) + 1.0*a_x/env.conf.acc + 1.0*a_v/env.conf.acc
        #        a_sign = phase + 1.0*a_x/env.conf.acc + 1.0*a_v/env.conf.acc
        a = 2 if a_sign > env.conf.acc else (0 if a_sign < -env.conf.acc else 1)
        acc = env.action_to_acc[a]
        # LOGGER.info(f"{i}({acc}). Phase:{phase}, a_x:{a_x}, a_v:{a_v}, a_sign:{a_sign}")
        obs, r, term, _trunc, _inf = env.step(a)
        assert isinstance(obs, tuple)
        assert isinstance(obs0, tuple)
        agent.update_q(obs0, a, r, terminated=term, s1=obs, r0=r0)
        LOGGER.info(f"{i}({acc}). obs:{obs}, reward:{r:2.4f}, dReward:{r - r0:2.4f}, Q:{agent.q_values[obs0]}")
        seq.append(a)
        r0 = r
        obs0 = obs

    # LOGGER.info(env.rewards)

    if show:
        env.show_plot()

    # Go through same steps again and calculate actions based on q-factor
    obs0, _ = env.reset(options={"init": False})
    r0 = env.reward
    df = agent.conf.discount_factor
    for i in range(50):
        assert isinstance(obs0, tuple)
        a = int(np.nanargmax(agent.q_values[obs0]))
        obs, r, term, _trunc, _inf = env.step(a)
        assert isinstance(obs, tuple)
        max_q = np.nanmax(agent.q_values[obs])
        # LOGGER.info(f"step {i}, a:{a}, -> {obs}. {r-r0:2.4f} + {df}* {max_q:2.4f} = {r-r0+df*max_q:2.4f}")
        LOGGER.info(
            f"step {i}, a:{a}->{obs}. {r * (1 - df):2.4f} + {df}* {max_q:2.4f} = {r * (1 - df) + df * max_q:2.4f}"
        )
        obs0 = obs
        r0 = r

    if show:
        env.show_plot()
        agent.dump_results(episodes=1, steps=steps, n_terminated=1, n_truncated=0)


def test_state_sequence2(crane: Callable[..., Crane], *, show: bool) -> None:
    """Go through an algorithmic action sequence and use that to investigate the q_agent."""

    env, agent = config_env_agent(
        crane,
        env_conf={"render_mode": "none", "start_speed": 2.0},
        file=Path(__file__).parent.resolve().parent / "models/q_sequence.json",
        use_file="r",
    )

    # fill the whole Q-table
    count = 1
    q_default = True  # expect default values in Q-table
    while q_default and count < 155:
        q_default = False
        obs0, _ = env.reset(options={"init": False})
        assert env.conf.reward_limit is not None
        r0 = -1.0
        while not q_default and r0 < env.conf.reward_limit:
            assert isinstance(obs0, tuple)
            _q = agent.q_values[obs0].copy()  # Q-value before update
            new_state = all(np.isnan(_a) or _a == agent.conf.q_default for _a in _q)
            action = int(env.action_space.sample() if new_state else np.argmax(_q))
            q_default = np.isnan(_q[action]) or _q[action] == agent.conf.q_default  # check whether action was new
            obs, reward, term, trunc, _ = env.step(action)  # take action and observe result
            if term or trunc:
                break
            assert isinstance(obs0, tuple)
            assert isinstance(obs, tuple)
            agent.update_q(obs0, action, reward, terminated=term, s1=obs, r0=r0)
            if q_default:
                assert isinstance(obs0, tuple)
                if action != np.argmax(_q):  # new action was not best
                    LOGGER.info(f"Filled in {action} => Q:{_q} -> {agent.q_values[obs0]}")
                    break  # do not follow this path further
                # ran a new action which turned out to be best
                LOGGER.info(f"Updated action {action} for obs {obs0}: Q:{_q} -> {agent.q_values[obs0]}, -> {obs}")
            r0 = reward
            obs0 = obs
        LOGGER.info(f"Iteration {count}. Reward:{reward}")
        count += 1
    agent.dump_results(filename=Path(__file__).parent.resolve().parent / "models/q_sequence_all.json")
    # agent.deterministic_episode()


def test_state2(crane: Callable[..., Crane], *, show: bool) -> None:
    """Set state and calculate reward."""

    def all_actions(
        pos: float = 0.0, speed: float = 0.0, direction: float = 0.0, w_speed: float = 2.0, *, details: bool = False
    ):
        """Set/reset a state and try all action possibilities."""
        env.reset(options={"init": True})
        obs, reward, _ = env.set_state(pos=pos, speed=speed, direction=np.radians(direction), w_speed=w_speed)
        # state0_info = env.get_state()
        max_d_reward = -float("inf")
        max_action = -1
        for a in range(3):
            env.reset(options={"init": True})
            o, r, _ = env.set_state(pos=pos, speed=speed, direction=np.radians(direction), w_speed=w_speed)
            state0 = env.get_state()
            assert np.allclose(o, obs)
            assert reward == r
            LOGGER.info(f"   prepared state: {env.get_state(as_x=False)}, reward:{r}")
            _obs, _reward, _term, _trunc, _ = env.step(a)
            LOGGER.info(f"   after step {a}: {env.get_state(as_x=False)}, reward:{_reward}")
            if _reward - reward > max_d_reward:
                max_d_reward = _reward - reward
                max_action = a
            if details:
                LOGGER.info(f"   a:{a} -> state: pos:{env.get_state()}, obs:{_obs}, reward:{_reward:2.3f}")
        LOGGER.info(
            f"Experiment. state:{state0}, obs:{obs}, reward:{reward:2.3f}. Max: {max_d_reward:2.3f}@ {max_action}"
        )

    env, _agent = config_env_agent(crane, env_conf={"start_speed": 2.0})
    #     env.set_state(pos=0.0, speed=0.0, direction=0.0, w_speed=0.0)
    #     for _i in range(10):
    #         assert np.allclose(env.get_state(), (0, 0, 0, 0))
    #         env.step(1)  # check that nothing moves
    #
    #     env.step(0)
    #     state = env.get_state()
    #     assert np.allclose(state, (-0.1, -0.1, 0.26384339641900634, 0.08442572592699621)), f"Found {state}"

    # all_actions(pos=0.0, speed=0.0, direction=3.0, w_speed=1.5)
    all_actions(pos=0.0, speed=0.0, direction=0.0, w_speed=2.0, details=True)
    # all_actions(pos=0.0, speed=0.0, direction=3.0, w_speed=0.0)


def test_update_q_values(crane: Callable[..., Crane], *, show: bool) -> None:
    env = AntiPendulumEnv(
        crane,
        conf=AntiPendulumConfig(
            start_speed=-1.0,
            render_mode="none",
            reward_limit=-0.05,
            reward_fac=RewardConfig.from_dict({"energy": 0.01, "positional": 0.01}),
            discrete="energy",
        ),
    )
    env.reset(options={"init": True})
    agent = QLearningAgent(env)
    env.set_state(pos=2.0, speed=0.0, direction=0.0, w_speed=0.0)
    env.step(1)  # neutral step

    obs, _ = env.reset()  # first reward is also available as self.env.reward
    # num_failed = 0

    for _i in range(1000):
        prev_reward = env.reward
        assert isinstance(obs, tuple)
        action = int(agent.get_action(obs))  # choose action (initially random, gradually more intelligent)
        next_obs, _reward, _terminated, _truncated, _ = env.step(action)  # take action and observe result
        reward = float(_reward)
        assert isinstance(next_obs, tuple)
        agent.update_q(obs, action, reward, terminated=False, s1=next_obs, r0=prev_reward)
        # Move to next state
        obs = next_obs
        # truncated = False

    LOGGER.info(f"REWARDS: {env.rewards}")


def test_config():
    """Test reading and writing config from/to file."""

    def do_tests(
        info: dict[str, Any], q_values: dict[tuple[int, ...], np.ndarray], *, only_general: bool = False
    ) -> QLearningAgent:
        assert info["environment"]["length"] == 10.0
        assert info["environment"]["seed"] == 1
        assert not info["environment"]["randomize_start"]
        assert info["environment"]["render_mode"] in ("none", "data"), f"Found {info['environment']['render_mode']}"
        assert isinstance(info["environment"]["reward_fac"], dict)
        assert isinstance(info["environment"]["reward_fac"], dict)
        assert isinstance(info["environment"]["discretization"]["angle"], list)
        assert info["q_agent"]["q_default"] is None
        for k, v in q_values.items():
            assert isinstance(k, tuple), f"Should be converted to tuple. Found {type(k)}"
            assert isinstance(v, np.ndarray), f"Should be converted to ndarray. Found {type(v)}"
            assert len(v) == 3

        env = AntiPendulumEnv(conf=AntiPendulumConfig(**info["environment"]))
        assert np.allclose(env.wire.origin, (0, 0, 10))
        assert np.allclose(env.wire.end, (0, 0, 0))
        assert isinstance(env.conf, AntiPendulumConfig)
        for k in asdict(env.conf):  # type: ignore[assignment]
            assert k in info["environment"], f"Configuration key {k} not found in 'environment'"
        info["q_agent"]["auto_run"] = 0
        info["q_agent"]["use_file"] = "r"
        agent = QLearningAgent(env, conf=QLearningConfig(**info["q_agent"]), q_values=q_values)  # type: ignore[arg-type]
        for k in asdict(agent.conf):  # type: ignore[assignment]
            assert k in info["q_agent"], f"Configuration key {k} not found in 'q_agent'"

        if not only_general:
            assert np.allclose(q_values[(6, 7, 5, 5, 5)], (0, 0, 0))
            assert next(iter(agent.q_values.keys())) == (6, 7, 5, 5, 5)
            assert np.allclose(next(iter(agent.q_values.values())), [0, 0, 0])

        return agent

    file = MODELS / "config_template.json"  # read the raw template
    assert file.exists(), f"File {file} not found"
    info, q_values = QLearningAgent.read_dumped(file)
    agent = do_tests(info, q_values)
    new_q = agent.q_values[(0, 0, 0, 0, 0)]
    assert all(np.isnan(x) for x in new_q), f"New values shall receive the default value. Found {new_q}"

    saved = Path(__file__).parent.resolve() / "test_working_directory/test_dump.json"
    agent.dump_results(
        filename=saved, episodes=9, steps=8, start_time=dt.datetime.now(dt.UTC), n_terminated=7, n_truncated=6
    )

    info, q_values = QLearningAgent.read_dumped(saved)
    agent = do_tests(info, q_values)
    assert info["q_agent"]["start_training"] == info["q_agent"]["end_training"]
    assert info["q_agent"]["episodes"] == 9
    assert info["q_agent"]["steps"] == 8
    assert info["q_agent"]["num_terminated"] == 9541 + 7
    assert info["q_agent"]["num_truncated"] == 10459 + 6
    assert all(np.isnan(x) for x in q_values[(0, 0, 0, 0, 0)]), f"Found {q_values[(0, 0, 0, 0, 0)]}"

    file = MODELS / "q_anti-pendulum1.json"  # test also another file
    assert file.exists(), f"File {file} not found"
    info, q_values = QLearningAgent.read_dumped(file)
    agent = do_tests(info, q_values, only_general=True)


def update_files():
    res_folder = Path(__file__).parent.resolve() / "test_working_directory"
    for file in MODELS.glob("*.json"):
        name = file.name
        if name.startswith("q_anti-pendulum"):
            LOGGER.info(f"File {name} ...")
            info, q_values = QLearningAgent.read_dumped(file)
            env = AntiPendulumEnv(conf=AntiPendulumConfig(**info["environment"]))
            agent = QLearningAgent(env, conf=QLearningConfig(**info["q_agent"]), q_values=q_values)
            agent.dump_results(res_folder / name)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import pytest

    from crane_controller.crane_factory import build_crane

    logging.basicConfig(level=logging.INFO)
    retcode = pytest.main(["-rP -s -v", __file__])
    assert retcode == 0, f"Return code {retcode}"
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")

    # test_levels(build_crane)
    test_intervals(build_crane)
    # test_smoke(build_crane, show=True)
    # test_q_analyse(build_crane, show=True)
    # test_discretization(build_crane, show=True, discretization='energy')
    # test_discretization(build_crane, show=True, discretization='phase')
    # test_state(build_crane, show=True)
    # test_state2(build_crane, show=True)
    # test_state_sequence1(build_crane, show=True)
    # test_state_sequence2(build_crane, show=True)
    # test_update_q_values(build_crane, show=True)
    # test_config()
    # update_files()
