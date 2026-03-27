from gymnasium.envs.registration import register

register(
    id="crane_controller/GridWorld-v0",
    entry_point="crane_controller.envs:GridWorldEnv",
)
