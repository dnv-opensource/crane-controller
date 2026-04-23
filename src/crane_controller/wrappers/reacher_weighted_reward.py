"""Gymnasium wrapper for weighted distance and control cost rewards."""

import gymnasium as gym


class ReacherRewardWrapper(gym.Wrapper[object, object, object, object]):
    """Gymnasium wrapper combining distance and control cost rewards."""

    def __init__(self, env: gym.Env[object, object], reward_dist_weight: float, reward_ctrl_weight: float) -> None:
        """Initialize the weighted reward wrapper.

        Parameters
        ----------
        env : gym.Env[object, object]
            The environment to wrap.
        reward_dist_weight : float
            Weight applied to the distance reward component.
        reward_ctrl_weight : float
            Weight applied to the control cost reward component.
        """
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, float]]:
        """Step the environment and return a weighted reward.

        Combines distance and control cost rewards from the info dict
        using the configured weights.

        Parameters
        ----------
        action : object
            Action to perform.

        Returns
        -------
        tuple[object, float, bool, bool, dict[str, float]]
            ``(observation, reward, terminated, truncated, info)``.
        """
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_dist_weight * info["reward_dist"] + self.reward_ctrl_weight * info["reward_ctrl"]
        return obs, reward, terminated, truncated, info
