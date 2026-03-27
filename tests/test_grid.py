# import gymnasium
# import gymnasium_env
# env = gymnasium.make('gymnasium_env/GridWorld-v0')

import numpy as np

from crane_controller.envs.grid_world import GridWorldEnv

grd = GridWorldEnv(render_mode="human", size=6)
grd.reset()
terminated = False
count = 0
while not terminated:
    count += 1
    observation, reward, terminated, res, info = grd.step(np.random.randint(0, 4))
print(f"Step count: {count}")
# print("Observation", observation)
# print("Reward", reward)
# print("Terminated", terminated)
# print("res", res)
# print("info", info)
