Crane Controller
================

AI-based control of crane systems using reinforcement learning, developed at DNV AS.

The primary goal is to solve the **anti-pendulum problem**: training an agent to dampen (or start)
the swing of a load hanging from a mobile crane, using only horizontal crane acceleration as the control input.

Environments
------------

``AntiPendulumEnv``
    The main environment. A mobile crane with a swinging load modelled via real crane physics
    (``py-crane`` library). The agent controls horizontal crane acceleration and must either
    start or stop the pendulum motion.

    - **Observation**: crane x-position, crane x-velocity, load polar angle, load x-velocity
    - **Actions**: Discrete(3) — accelerate left / coast / accelerate right
    - **Modes**: *start* (build pendulum energy) or *stop* (dampen swing)

``ControlledCraneEnv``
    A more general mobile crane environment for future work.

Algorithms
----------

Three RL algorithms are implemented, each as a self-contained agent class:

- **PPO** (``ppo_agent.py``) — Proximal Policy Optimization via ``stable-baselines3``. Supports
  vectorized environments for faster training. Models saved as ``.zip`` files.

- **Q-Learning** (``q_agent.py``) — Tabular Q-learning with epsilon-greedy exploration.
  Uses a discretized observation space. Q-tables saved/loaded as JSON for incremental training.

- **REINFORCE** (``reinforce_agent.py``) — Policy gradient with a PyTorch neural network policy.
  Two-layer network (16 → 32 units, Tanh) outputting a Normal distribution over actions.

- **AlgorithmAgent** (``algorithm.py``) — Brute-force search over all 81 handcoded strategies
  (3\ :sup:`4` combinations). Useful as a baseline.

Wrappers
--------

Generic Gymnasium wrappers (from the Farama Foundation examples) are included for reference:

- ``ClipReward`` — clips immediate rewards to a valid range
- ``DiscreteActions`` — restricts the action space to a finite subset
- ``RelativePosition`` — computes relative position between agent and target
- ``ReacherRewardWrapper`` — weights multiple reward terms

Learning Examples
-----------------

Two classic Gymnasium environments were used as stepping stones when developing this project:

- **GridWorldEnv** — minimal grid navigation, ideal for learning the Gymnasium API.
  See the `environment creation tutorial <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/>`_
  and the `Gymnasium examples repo <https://github.com/Farama-Foundation/gymnasium-env-template>`_.
- **CartPoleEnv** — cart-pole balancing, useful for verifying RL algorithms before applying them
  to the crane. Available via ``gymnasium.make("CartPole-v1")``.

Installation
------------

.. code-block:: shell

   cd crane-controller
   pip install -e .

Running
-------

Install dependencies and run the test suite with ``uv``:

.. code-block:: shell

   uv run pytest tests/ -v

Test files are organised by algorithm:

- ``tests/test_crane_pendulum.py`` — environment, Q-learning, and algorithm tests
- ``tests/test_ppo.py`` — PPO pipeline smoke test (``test_monitor``)

Tests are suitable for CI/CD — no plot windows are produced.

Training
--------

**PPO:**

.. code-block:: shell

   uv run python scripts/train_ppo.py

Key options:

- ``--steps N`` — total training timesteps (default: 100 000)
- ``--n-envs N`` — number of parallel environments (default: 4)
- ``--save-path PATH`` — where to write the trained model (default: ``models/ppo_AntiPendulumEnv.zip``)
- ``--dry-run`` — run 1 000 steps with a live reward-tracking plot and no model saved

**Q-learning:**

.. code-block:: shell

   uv run python scripts/train_q.py

Key options:

- ``--episodes N`` — total training episodes (default: 10 000)
- ``--v0 F`` — initial crane speed; negative = stop mode, positive = start mode (default: ``-1.0``)
- ``--reward-limit F`` — per-episode termination threshold (default: ``-0.05``)
- ``--save-path PATH`` — where to write the Q-table (default: ``models/q_AntiPendulumEnv.json``)
- ``--trained PATH`` — continue training from an existing Q-table JSON
- ``--intervals N`` — run interval training: N rounds of 10 episodes each
- ``--dry-run`` — run 50 episodes with a reward plot and no model saved

Playing
-------

Run a trained agent visually. Both scripts accept ``--render-mode`` with the following options:

- ``plot`` — 4-panel figure per episode (load angle, crane position/speed, rewards)
- ``play-back`` — animated crane trajectory after each episode
- ``reward-tracking`` — live reward line plot updating every step

**PPO** (default render-mode: ``play-back``):

.. code-block:: shell

   uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip
   uv run python scripts/play_ppo.py --model-path models/ppo_AntiPendulumEnv.zip --render-mode plot --episodes 3

**Q-learning** (default render-mode: ``plot``):

.. code-block:: shell

   uv run python scripts/play_q.py --model-path models/q_AntiPendulumEnv.json
   uv run python scripts/play_q.py --model-path tests/anti-pendulum.json --render-mode play-back --episodes 3

Analysing
---------

Inspect a trained Q-table without running the environment:

.. code-block:: shell

   uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json

Prints per-pos/speed average Q-values for a quick sanity check. To drill into
specific states, use ``--obs`` with 5 integers (use ``-1`` as a wildcard):

.. code-block:: shell

   uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json --obs -1 0 0 -1 -1

The five observation dimensions are: ``[energy, pos, speed, distance, sector]``.

Contributing
------------

- Fork this repository
- Clone your fork
- Set up pre-commit hooks: ``pre-commit install``
