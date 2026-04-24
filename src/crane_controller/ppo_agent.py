from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

plt.rcParams["figure.figsize"] = (10, 5)


class ProximalPolicyOptimizationAgent:
    """`PPO algorithm <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html`_.

    Initializes an agent that learns a policy via PPO algorithm to solve the task at hand.

    PPO agents can be saved as zip file and re-loaded to avoid re-training.
    VecNormalize statistics are saved alongside the model as `<name>_vecnorm.pkl`.

    Args:
        env (gym.Env): the environment the agent is acting on.
        n_envs (int) = 4: The number of environments to used during training.
          n_envs=0 signals that the trained agent should be loaded from file.
        env_kwargs (dict): Optional possibility to provide additional kwargs for environment
        trained (str|Path): Optional file name for saving/loading the trained agent. Required for n_envs=0
    """

    def __init__(
        self,
        env: gym.Env,
        n_envs: int = 4,
        env_kwargs: dict[str, Any] | None = None,
        trained: tuple[str, bool] | None = None,
    ):
        self.trained = trained
<<<<<<< HEAD
        if env_kwargs is None:
            self.env = env()  # type: ignore[operator]  ## the object is callable! (__init__())
        else:
            self.env = env(**env_kwargs)  # type: ignore[operator]  ## the object is callable! (__init__())
        _n_envs = n_envs = 1 if n_envs <= 0 else n_envs
        self.vec_env = make_vec_env(env_id=env, n_envs=_n_envs, env_kwargs=env_kwargs)  # type: ignore ## should be correct
        if n_envs <= 0:
            assert self.trained is not None, "When no training is specified a saved model should be provided"
            self.model = PPO.load(self.trained[0])
        elif n_envs == 1:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            if trained is not None:
                self.trained = (trained[0], trained[1])
        else:
            self.model = PPO("MlpPolicy", self.vec_env)
=======
        inference_only = n_envs <= 0
        _n_envs = 1 if inference_only else n_envs

        raw_vec_env = make_vec_env(env_id=env, n_envs=_n_envs, env_kwargs=env_kwargs)  # type: ignore

        if inference_only:
            assert trained is not None, "When no training is specified a saved model should be provided"
            stats_path = self._stats_path(trained[0])
            if stats_path.exists():
                self.vec_env = VecNormalize.load(str(stats_path), raw_vec_env)
            else:
                self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.model = PPO.load(trained[0], env=self.vec_env)
        else:
            self.vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=True)
            if _n_envs == 1:
                self.model = PPO("MlpPolicy", self.vec_env, verbose=1)
            else:
                self.model = PPO("MlpPolicy", self.vec_env)
>>>>>>> dd5de9aa04b842647a963b8d0ecba489aff15087
            self.trained = (
                trained[0] if trained is not None else f"ppo_{env.__name__}",  # type: ignore[attr-defined]
                False if trained is None else trained[1],
            )

        # Single unwrapped env for do_one_episode/evaluate without reconstructing a new crane.
        self.env = self.vec_env.venv.envs[0]  # type: ignore[attr-defined]

    @staticmethod
    def _stats_path(model_path: str) -> Path:
        p = Path(model_path)
        return p.parent / f"{p.stem}_vecnorm.pkl"

    def do_training(self, total_timesteps: int = 25000, progress_bar: bool = True):
        self.model.learn(total_timesteps, progress_bar=progress_bar)
        if self.trained is not None and self.trained[1] and self.env.render_mode not in ("play-back",):
            self.model.save(self.trained[0])
            self.vec_env.save(str(self._stats_path(self.trained[0])))

    def evaluate(self, n_episodes: int = 10):
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=n_episodes)
        self.vec_env.training = True
        self.vec_env.norm_reward = True
        print(f"Mean:{mean_reward}, stdev:{std_reward}")

    def do_one_episode(self, seed: int = 1):
        """Do one episode using the trained normalizer for observations."""
        obs, info = self.env.reset(seed=seed)
        terminated = truncated = False
        while not terminated and not truncated:
            norm_obs = self.vec_env.normalize_obs(obs)
            action, _states = self.model.predict(norm_obs, deterministic=True)
            obs, rewards, terminated, truncated, info = self.env.step(action)
        self.env.render()


# class TrainingMonitor(BaseCallback):
#     def __init__(self, verbose:int=0):
#         super().__init__(verbose)
#
#     def _on_step(self):
#         """Called at every step of the training."""
#
#         return True
