import gymnasium as gym


class ReacherRewardWrapper(gym.Wrapper[object, object, object, object]):
    def __init__(self, env: gym.Env[object, object], reward_dist_weight: float, reward_ctrl_weight: float) -> None:
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, float]]:
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_dist_weight * info["reward_dist"] + self.reward_ctrl_weight * info["reward_ctrl"]
        return obs, reward, terminated, truncated, info
