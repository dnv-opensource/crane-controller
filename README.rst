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

Some tests produce plots or animations. These are suppressed by default (suitable for CI/CD).
To enable visual output locally, pass ``--show True``:

.. code-block:: shell

   uv run pytest tests/test_crane_pendulum.py::test_init -v --show

Contributing
------------

- Fork this repository
- Clone your fork
- Set up pre-commit hooks: ``pre-commit install``
