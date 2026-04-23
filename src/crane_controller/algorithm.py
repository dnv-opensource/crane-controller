from itertools import product

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm  # Progress bar


class AlgorithmAgent(object):
    """Agent for algorithmic control of a (simple) environment.

    Args:
        env (gym.Env): The Environment (class) to be controlled. Need .reset() and .step() functions.
    """

    envs = ("AntiPendulumEnv",)

    def __init__(
        self,
        env: gym.Env,
    ):
        self.env = env
        assert type(self.env).__name__ in AlgorithmAgent.envs, f"Environment {type(self.env).__name__} not listed."

        # Track learning progress
        self.training_error: list[float] = []
        self.combination: tuple[int, ...]

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action based on the current observation of load position (obs[1]) and speed (obs(2)).
        The algorithmic strategy is coded as self.strategy and the observation slots 0,3,4 are ignored.

        Returns
        -------
            action: an allowed action from the action space
        """
        # print("OBS", obs)
        if obs == (0, 0, 0, 0, 0):  # in start mode. Random push to get started.
            return (0, 2)[np.random.randint(0, 2)]  # random 0 or 2
        else:  # see the reward of all strategies at end of file. 0202 is optimal
            if obs[1] == 0 and obs[2] == 0:
                return self.strategy[0]
            elif obs[1] == 0 and obs[2] == 1:
                return self.strategy[1]
            elif obs[1] == 1 and obs[2] == 0:
                return self.strategy[2]
            elif obs[1] == 1 and obs[2] == 1:
                return self.strategy[3]
            else:
                raise ValueError("There should not be other choices {obs}") from None

    def do_strategies(self, max_steps: int = 5000):
        """Go through all strategies, where
            pos=-, speed=- => strategy[0]
            pos=-, speed=+ => strategy[1]
            pos=+, speed=- => strategy[2]
            pos=+, speed=+ => strategy[3]
            with pos=obs[1], speed=obs[2]
        Observations 0, 3 and 4 are ignored.
        """
        res = []
        for self.strategy in product(range(3), range(3), range(3), range(3)):
            obs, info = self.env.reset()
            terminated, truncated = (False, False)
            steps = 0
            reward = 0.0
            while not terminated and not truncated:
                steps += 1
                action = self.get_action(obs)  # choose action (initially random, then deterministic)
                obs, _reward, terminated, truncated, info = self.env.step(action)  # take action and observe result
                reward = float(_reward)
                if steps > max_steps:
                    truncated = True
            res.append(reward)
        if not self.env.render_mode == 'none':
            for i, self.strategy in enumerate(product(range(3), range(3), range(3), range(3))):
                print(f"{i}. strategy {self.strategy}: {res[i]}")

    def do_episodes(self, n_episodes: int = 1000, show: int = 0, max_steps: int = 1000):
        """Run episodes."""
        for _episode in tqdm(range(n_episodes)):
            # Start a new episode
            obs, info = self.env.reset()
            terminated, truncated = (False, False)
            steps = 0
            reward = 0.0

            while not terminated and not truncated:
                steps += 1
                action = self.get_action(obs)  # choose action (initially random, then deterministic)
                # reward0 = reward
                obs, _reward, terminated, truncated, info = self.env.step(action)  # take action and observe result
                reward = float(_reward)
                if show == 2:
                    self.analyse_episode()
                if steps > max_steps:
                    truncated = True
            print(f"strategy {self.strategy}: {reward}")
        self.env.reset()  # reset at end so that rendering is performed properly

        if show == 1:
            self.analyse_training()

    def analyse_training(self, window: int = 500):

        def get_moving_avgs(arr, window, convolution_mode):
            """Compute moving average to smooth noisy data."""
            return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window

        # Smooth over the given episode window
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

        lengths = [row[0] for row in self.env.reward_stats]  # type: ignore ## reward_stats exist
        rewards = [row[1] for row in self.env.reward_stats]  # type: ignore ## reward_stats exist

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = get_moving_avgs(rewards, int(window / 10), "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = get_moving_avgs(lengths, int(window / 10), "valid")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = get_moving_avgs(self.training_error, window, "same")
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def analyse_episode(self, window: int = 100):

        def get_moving_avgs(arr, window, convolution_mode):
            """Compute moving average to smooth noisy data."""
            return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window

        # Smooth over the given episode window
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = get_moving_avgs(self.env.rewards, window, "valid")  # type: ignore ## rewards exist!
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[1].set_title("Training Error")
        training_error_moving_average = get_moving_avgs(self.training_error, window, "same")
        axs[1].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[1].set_ylabel("Temporal Difference Error")
        axs[1].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

    def test_agent(self, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward: float = 0.0
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated

            total_rewards.append(episode_reward)

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        msg = f"Test Results over {num_episodes} episodes:\n"
        msg += f"Win Rate: {win_rate:.1%}\n"
        msg += f"Average Reward: {average_reward:.3f}\n"
        msg += f"Standard Deviation: {np.std(total_rewards):.3f}\n"
        return msg


# All strategies in 'start pendulum' mode: (0,2,0,2) is best
# 0. strategy (0, 0, 0, 0): 0.0029283626102375496
# 1. strategy (0, 0, 0, 1): 0.0029283626102375496
# 2. strategy (0, 0, 0, 2): 0.0029283626102375496
# 3. strategy (0, 0, 1, 0): 0.0029283626102375496
# 4. strategy (0, 0, 1, 1): 0.0029283626102375496
# 5. strategy (0, 0, 1, 2): 0.0029283626102375496
# 6. strategy (0, 0, 2, 0): 0.0029283626102375496
# 7. strategy (0, 0, 2, 1): 0.0029283626102375496
# 8. strategy (0, 0, 2, 2): 0.0029283626102375496
# 9. strategy (0, 1, 0, 0): 0.2694186364324193
# 10. strategy (0, 1, 0, 1): 0.7667219096156876
# 11. strategy (0, 1, 0, 2): 1.5973232641146864
# 12. strategy (0, 1, 1, 0): 0.06909999105717882
# 13. strategy (0, 1, 1, 1): 0.2867806368007592
# 14. strategy (0, 1, 1, 2): 0.8128224830809284
# 15. strategy (0, 1, 2, 0): 0.0015294416143673234
# 16. strategy (0, 1, 2, 1): 0.0003055863185055837
# 17. strategy (0, 1, 2, 2): 0.2654610874863169
# 18. strategy (0, 2, 0, 0): 1.0011851068466873
# 19. strategy (0, 2, 0, 1): 1.8488186600898455
# 20. strategy (0, 2, 0, 2): 3.0710648835742944
# 21. strategy (0, 2, 1, 0): 0.29544831864231724
# 22. strategy (0, 2, 1, 1): 0.878918205429762
# 23. strategy (0, 2, 1, 2): 1.6909322331854766
# 24. strategy (0, 2, 2, 0): 2.27131305851893e-05
# 25. strategy (0, 2, 2, 1): 0.17387145806824872
# 26. strategy (0, 2, 2, 2): 0.6761989819801578
# 27. strategy (1, 0, 0, 0): 9.81000081168304e-10
# 28. strategy (1, 0, 0, 1): 9.81000081168304e-10
# 29. strategy (1, 0, 0, 2): 9.81000081168304e-10
# 30. strategy (1, 0, 1, 0): 9.81000081168304e-10
# 31. strategy (1, 0, 1, 1): 9.81000081168304e-10
# 32. strategy (1, 0, 1, 2): 9.81000081168304e-10
# 33. strategy (1, 0, 2, 0): 9.81000081168304e-10
# 34. strategy (1, 0, 2, 1): 9.81000081168304e-10
# 35. strategy (1, 0, 2, 2): 9.81000081168304e-10
# 36. strategy (1, 1, 0, 0): 9.81000081168304e-10
# 37. strategy (1, 1, 0, 1): 9.81000081168304e-10
# 38. strategy (1, 1, 0, 2): 9.81000081168304e-10
# 39. strategy (1, 1, 1, 0): 9.81000081168304e-10
# 40. strategy (1, 1, 1, 1): 9.81000081168304e-10
# 41. strategy (1, 1, 1, 2): 9.81000081168304e-10
# 42. strategy (1, 1, 2, 0): 9.81000081168304e-10
# 43. strategy (1, 1, 2, 1): 9.81000081168304e-10
# 44. strategy (1, 1, 2, 2): 9.81000081168304e-10
# 45. strategy (1, 2, 0, 0): 9.81000081168304e-10
# 46. strategy (1, 2, 0, 1): 9.81000081168304e-10
# 47. strategy (1, 2, 0, 2): 9.81000081168304e-10
# 48. strategy (1, 2, 1, 0): 9.81000081168304e-10
# 49. strategy (1, 2, 1, 1): 9.81000081168304e-10
# 50. strategy (1, 2, 1, 2): 9.81000081168304e-10
# 51. strategy (1, 2, 2, 0): 9.81000081168304e-10
# 52. strategy (1, 2, 2, 1): 9.81000081168304e-10
# 53. strategy (1, 2, 2, 2): 9.81000081168304e-10
# 54. strategy (2, 0, 0, 0): 3.013176875874262e-07
# 55. strategy (2, 0, 0, 1): 5.142686678248059e-07
# 56. strategy (2, 0, 0, 2): 2.27131305851893e-05
# 57. strategy (2, 0, 1, 0): 1.6394328338813897e-08
# 58. strategy (2, 0, 1, 1): 2.199734202829083e-07
# 59. strategy (2, 0, 1, 2): 0.0015294416143673234
# 60. strategy (2, 0, 2, 0): 2.8681185829477665e-07
# 61. strategy (2, 0, 2, 1): 0.0001949003756618035
# 62. strategy (2, 0, 2, 2): 0.0029283626102375496
# 63. strategy (2, 1, 0, 0): 1.7871147946675603e-08
# 64. strategy (2, 1, 0, 1): 8.379429879100584e-06
# 65. strategy (2, 1, 0, 2): 0.29544831864231724
# 66. strategy (2, 1, 1, 0): 1.6394328338813897e-08
# 67. strategy (2, 1, 1, 1): 2.0798200613084902e-08
# 68. strategy (2, 1, 1, 2): 0.06909999105717882
# 69. strategy (2, 1, 2, 0): 1.6041830524154352e-08
# 70. strategy (2, 1, 2, 1): 0.0001949003756618035
# 71. strategy (2, 1, 2, 2): 0.0029283626102375496
# 72. strategy (2, 2, 0, 0): 0.00027041109992226065
# 73. strategy (2, 2, 0, 1): 0.19319407789380985
# 74. strategy (2, 2, 0, 2): 1.0011851068466873
# 75. strategy (2, 2, 1, 0): 1.6394328338813897e-08
# 76. strategy (2, 2, 1, 1): 2.0590036981571137e-05
# 77. strategy (2, 2, 1, 2): 0.2694186364324193
# 78. strategy (2, 2, 2, 0): 3.01372394973002e-07
# 79. strategy (2, 2, 2, 1): 0.0001949003756618035
# 80. strategy (2, 2, 2, 2): 0.0029283626102375496

# All strategys in 'stop pendulum' mode: strategy 2,1,1,0 is best
# 0. strategy (0, 0, 0, 0): -5.756935736071321
# 1. strategy (0, 0, 0, 1): -4.066021934129539
# 2. strategy (0, 0, 0, 2): -3.052879906871171
# 3. strategy (0, 0, 1, 0): -3.745900785357636
# 4. strategy (0, 0, 1, 1): -2.3178038902966893
# 5. strategy (0, 0, 1, 2): -1.3830888744744634
# 6. strategy (0, 0, 2, 0): -0.0022892926121056336 # strange result because not all observations included
# 7. strategy (0, 0, 2, 1): -0.5751057940340446
# 8. strategy (0, 0, 2, 2): -0.2798994062043238
# 9. strategy (0, 1, 0, 0): -4.318889338943463
# 10. strategy (0, 1, 0, 1): -3.450950316292415
# 11. strategy (0, 1, 0, 2): -3.178006389859391
# 12. strategy (0, 1, 1, 0): -2.1416805037007554
# 13. strategy (0, 1, 1, 1): -1.495101991479309
# 14. strategy (0, 1, 1, 2): -1.4640521700253069
# 15. strategy (0, 1, 2, 0): -0.5526520140588366
# 16. strategy (0, 1, 2, 1): -0.2737608972773078
# 17. strategy (0, 1, 2, 2): -1.102045451811489
# 18. strategy (0, 2, 0, 0): -4.368317658860986
# 19. strategy (0, 2, 0, 1): -4.283982581777573
# 20. strategy (0, 2, 0, 2): -4.655705963788677
# 21. strategy (0, 2, 1, 0): -1.6631611738173675
# 22. strategy (0, 2, 1, 1): -1.8152643486607083
# 23. strategy (0, 2, 1, 2): -3.3392553450248332
# 24. strategy (0, 2, 2, 0): -0.2715995131013247
# 25. strategy (0, 2, 2, 1): -1.2546358963697362
# 26. strategy (0, 2, 2, 2): -3.0926280953791307
# 27. strategy (1, 0, 0, 0): -3.45143095917313
# 28. strategy (1, 0, 0, 1): -1.9365622554281625
# 29. strategy (1, 0, 0, 2): -1.1784100889046005
# 30. strategy (1, 0, 1, 0): -1.9492872721025705
# 31. strategy (1, 0, 1, 1): -0.9705359179569177
# 32. strategy (1, 0, 1, 2): -0.1412006663875676
# 33. strategy (1, 0, 2, 0): -0.26094222167179476
# 34. strategy (1, 0, 2, 1): -0.40655914218667866
# 35. strategy (1, 0, 2, 2): -0.31954175021069137
# 36. strategy (1, 1, 0, 0): -1.9064546436037717
# 37. strategy (1, 1, 0, 1): -1.4756953654913474
# 38. strategy (1, 1, 0, 2): -1.7465670427177389
# 39. strategy (1, 1, 1, 0): -0.5813378897833512
# 40. strategy (1, 1, 1, 1): -0.187584307009098
# 41. strategy (1, 1, 1, 2): -1.4730236417872544
# 42. strategy (1, 1, 2, 0): -0.47737519400369477
# 43. strategy (1, 1, 2, 1): -0.7212614317050706
# 44. strategy (1, 1, 2, 2): -1.979141641362017
# 45. strategy (1, 2, 0, 0): -1.8444357492779788
# 46. strategy (1, 2, 0, 1): -2.4258915452461363
# 47. strategy (1, 2, 0, 2): -4.436142229161793
# 48. strategy (1, 2, 1, 0): -0.2812242582237298
# 49. strategy (1, 2, 1, 1): -1.5430020419281996
# 50. strategy (1, 2, 1, 2): -3.6813170108849604
# 51. strategy (1, 2, 2, 0): -1.0923663290895256
# 52. strategy (1, 2, 2, 1): -2.0038580659537395
# 53. strategy (1, 2, 2, 2): -4.211919920954667
# 54. strategy (2, 0, 0, 0): -0.841915432506557
# 55. strategy (2, 0, 0, 1): -0.8474143744076379
# 56. strategy (2, 0, 0, 2): -0.1915824055013108
# 57. strategy (2, 0, 1, 0): -0.4078734167105293
# 58. strategy (2, 0, 1, 1): -0.29909938378231893
# 59. strategy (2, 0, 1, 2): -0.3816636520536118
# 60. strategy (2, 0, 2, 0): -0.20863377211083375
# 61. strategy (2, 0, 2, 1): -7.163602117059054e-05
# 62. strategy (2, 0, 2, 2): -0.0007268116959459297
# 63. strategy (2, 1, 0, 0): -0.48629832571515763
# 64. strategy (2, 1, 0, 1): -0.27792625615038435
# 65. strategy (2, 1, 0, 2): -1.76539292192857
# 66. strategy (2, 1, 1, 0): -0.05789066981365632
# 67. strategy (2, 1, 1, 1): -0.6608276604925473
# 68. strategy (2, 1, 1, 2): -2.181839058157101
# 69. strategy (2, 1, 2, 0): -1.43608876371599
# 70. strategy (2, 1, 2, 1): -1.6068659409491288
# 71. strategy (2, 1, 2, 2): -3.489277759754874
# 72. strategy (2, 2, 0, 0): -0.49476264714547075
# 73. strategy (2, 2, 0, 1): -1.911043225625798
# 74. strategy (2, 2, 0, 2): -4.4331939676932475
# 75. strategy (2, 2, 1, 0): -0.7500121505799353
# 76. strategy (2, 2, 1, 1): -2.06047155494332
# 77. strategy (2, 2, 1, 2): -4.448341636143389
# 78. strategy (2, 2, 2, 0): -1.3092085180636048
# 79. strategy (2, 2, 2, 1): -3.6692231237301893
# 80. strategy (2, 2, 2, 2): -5.84894352095235
