from __future__ import annotations

import logging
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.distributions.normal import Normal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

plt.rcParams["figure.figsize"] = (10, 5)


class PolicyNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int) -> None:
        """Initialize a neural network that estimates the mean and standard deviation.

         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, return the mean and standard deviation.

         of a normal distribution from which an action is sampled.

        Args:
            x: Observation from the environment

        Returns
        -------
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(
        self,
        env: gym.Env[object, object],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        """Initialize an agent that learns a policy via REINFORCE algorithm [1].

        to solve the task at hand (Inverted Pendulum v4).

        Args:
            env (gym.Env): the environment to be trained
            learning_rate (float) = 1e-4: Learning rate for policy optimization
            gamma (float) = 0.99: Discount factor
            eps : float = 1e-6: small number for mathematical stability
        """
        self.env = env
        assert self.env.observation_space.shape is not None
        self.obs_space_dims = self.env.observation_space.shape[0]  # Observation-space of environment
        assert self.env.action_space.shape is not None
        logger.error("Check the code for deciding whether an action space is discrete or continuous!")
        if len(self.env.action_space.shape) == 1:  # discrete space
            self.action_space_dims = self.env.action_space.shape[0]
        else:  # continuous space
            self.action_space_dims = self.env.action_space.shape[0]  # Action-space of environment

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps

    def reset(self) -> None:
        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNetwork(self.obs_space_dims, self.action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> int | np.ndarray:
        """Return an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns
        -------
            action: Action to be performed
        """
        state_tensor = torch.from_numpy(state.astype(np.float32, copy=False))
        action_means, action_stddevs = self.net(state_tensor)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        self.probs.append(prob)
        action = action.numpy()

        if self.env.action_space.shape is not None and not len(self.env.action_space.shape):  # discrete space
            return int(action[0])
        return action

    def update(self) -> None:
        """Update the policy network's weights."""
        running_g = 0
        gs: list[float] = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for reward in self.rewards[::-1]:
            running_g = reward + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(self.probs).squeeze()

        # Update the loss with the mean log probability and deltas
        # Now, we compute the correct total loss by taking the sum of the element-wise products.
        loss = -torch.sum(log_probs * deltas)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

    def do_training(self, num_episodes: int = 5000) -> list[list[float]]:
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(self.env, 50)  # Records episode-reward

        rewards_over_seeds: list[list[float]] = []

        for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
            # set seed
            torch.manual_seed(seed)
            random.seed(seed)

            # Reinitialize agent every seed
            self.reset()
            reward_over_episodes: list[float] = []

            for episode in range(num_episodes):
                # gymnasium v26 requires users to set seed while resetting the environment
                obs_arr, _ = wrapped_env.reset(seed=seed)

                done = False
                while not done:
                    action = self.sample_action(np.asarray(obs_arr))

                    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                    # These represent the next observation, the reward from the step,
                    # if the episode is terminated, if the episode is truncated and
                    # additional info from the step
                    obs_arr, reward, terminated, truncated, _ = wrapped_env.step(action)
                    self.rewards.append(reward)

                    # End the episode when either truncated or terminated is true
                    #  - truncated: The episode duration reaches max number of timesteps
                    #  - terminated: Any of the state space values is no longer finite.
                    #
                    done = terminated or truncated

                reward_over_episodes.append(wrapped_env.return_queue[-1])
                self.update()

                if episode % 1000 == 0:
                    avg_reward = int(np.mean(wrapped_env.return_queue))
                    logger.info("Episode: %s Average Reward: %s", episode, avg_reward)

            rewards_over_seeds.append(reward_over_episodes)

        return rewards_over_seeds

    def plot_learning_curve(self, rewards_over_seeds: list[list[float]]) -> None:
        df1 = pd.DataFrame(rewards_over_seeds).melt()
        df1 = df1.rename(columns={"variable": "episodes", "value": "reward"})
        sns.set(style="darkgrid", context="talk", palette="rainbow")
        sns.lineplot(x="episodes", y="reward", data=df1).set(title=f"REINFORCE for {type(self.env).__name__}")
        plt.show()
