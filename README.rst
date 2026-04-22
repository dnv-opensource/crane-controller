|pypi| |versions| |license| |ci| |docs|

Introduction
============

The package provides AI-based control of a crane system using reinforcement learning, developed at DNV AS.

The primary goal is to solve the **anti-pendulum problem**: training an agent to dampen (or start)
the swing of a load hanging from a mobile crane, using only horizontal crane acceleration as the control input.

Getting Started
---------------

Environments
^^^^^^^^^^^^

``AntiPendulumEnv``
    The main environment. A mobile crane with a swinging load modelled via real crane physics
    (``crane-controller`` library). The agent controls horizontal crane acceleration and must either
    start or stop the pendulum motion.

    - **Observation**: crane x-position, crane x-velocity, load polar angle, load x-velocity
    - **Actions**: Discrete(3) — accelerate left / coast / accelerate right
    - **Modes**: *start* (build pendulum energy) or *stop* (dampen swing)

``ControlledCraneEnv``
    A more general mobile crane environment for future work.

Algorithms
^^^^^^^^^^

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
^^^^^^^^

Generic Gymnasium wrappers (from the Farama Foundation examples) are included for reference:

- ``ClipReward`` — clips immediate rewards to a valid range
- ``DiscreteActions`` — restricts the action space to a finite subset
- ``RelativePosition`` — computes relative position between agent and target
- ``ReacherRewardWrapper`` — weights multiple reward terms

Learning Examples
^^^^^^^^^^^^^^^^^

Two classic Gymnasium environments were used as stepping stones when developing this project:

- **GridWorldEnv** — minimal grid navigation, ideal for learning the Gymnasium API.
  See the `environment creation tutorial <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/>`_
  and the `Gymnasium examples repo <https://github.com/Farama-Foundation/gymnasium-env-template>`_.
- **CartPoleEnv** — cart-pole balancing, useful for verifying RL algorithms before applying them
  to the crane. Available via ``gymnasium.make("CartPole-v1")``.

Installation
------------

.. code-block:: shell

   pip install crane-controller


Usage
-----

Running
^^^^^^^

Install dependencies and run the test suite with ``uv``:

.. code-block:: shell

   uv run pytest tests/ -v

Test files are organised by algorithm:

- ``tests/test_crane_pendulum.py`` — environment, Q-learning, and algorithm tests
- ``tests/test_ppo.py`` — PPO pipeline smoke test (``test_monitor``)

Tests are suitable for CI/CD — no plot windows are produced.

Training
^^^^^^^^

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
^^^^^^^

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
^^^^^^^^^

Inspect a trained Q-table without running the environment:

.. code-block:: shell

   uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json

Prints per-pos/speed average Q-values for a quick sanity check. To drill into
specific states, use ``--obs`` with 5 integers (use ``-1`` as a wildcard):

.. code-block:: shell

   uv run python scripts/analyse_q.py --model-path tests/anti-pendulum.json --obs -1 0 0 -1 -1

The five observation dimensions are: ``[energy, pos, speed, distance, sector]``.

Development Setup
-----------------

1. Install uv
^^^^^^^^^^^^^
This project uses `uv` as package manager.

If you haven't already, install `uv <https://docs.astral.sh/uv/>`_, preferably using it's `"Standalone installer" <https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2/>`_ method:

..on Windows:

.. code:: sh

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

..on MacOS and Linux:

.. code:: sh

   curl -LsSf https://astral.sh/uv/install.sh | sh

(see `docs.astral.sh/uv <https://docs.astral.sh/uv/getting-started/installation//>`_ for all / alternative installation methods.)

Once installed, you can update `uv` to its latest version, anytime, by running:

``uv self update``

2. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^
Clone the crane-controller repository into your local development directory:

.. code:: sh

   git clone https://github.com/dnv-opensource/crane-controller path/to/your/dev/crane-controller

Change into the project directory after cloning:

.. code:: sh

   cd crane-controller

3. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^
Run ``uv sync -U`` to create a virtual environment and install all project dependencies into it:

.. code:: sh

   uv sync -U

..

   **Note**: Using ``--no-dev`` will omit installing development
   dependencies.

   **Explanation**: The ``-U`` option stands for ``--update``. It forces
   ``uv`` to fetch and install the latest versions of all dependencies,
   ensuring that your environment is up-to-date.

..

   **Note**: ``uv`` will create a new virtual environment called
   ``.venv`` in the project root directory when running ``uv sync -U``
   the first time. Optionally, you can create your own virtual
   environment using e.g. ``uv venv``, before running ``uv sync -U``.

4. (Optional) Install CUDA support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run ``uv sync -U`` with option ``--extra cuda`` to in addition install
torch with CUDA support:

.. code:: sh

   uv sync -U --extra cuda

Alternatively, you can manually install torch with CUDA support. *Note
1*: Do this preferably *after* running ``uv sync -U``. That way you
ensure a virtual environment exists, which is a prerequisite before you
install torch with CUDA support using below ``uv pip install`` command.

To manually install torch with CUDA support, generate a
``uv pip install`` command matching your local machine’s operating
system using the wizard on the official `PyTorch
website <https://pytorch.org/get-started/locally/>`__. *Note*: As we use
``uv`` as package manager, remember to replace ``pip`` in the command
generated by the wizard with ``uv pip``.

If you are on Windows, the resulting ``uv pip install`` command will
most likely look something like this:

.. code:: sh

   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

*Hint:* If you are unsure which cuda version to indicate in above
``uv pip install .. /cuXXX`` command, you can use the shell command
``nvidia-smi`` on your local system to find out the cuda version
supported by the current graphics driver installed on your system. When
then generating the ``uv pip install`` command with the wizard from the
`PyTorch website <https://pytorch.org/get-started/locally/>`__, select
the cuda version that matches the major version of what your graphics
driver supports (major version must match, minor version may deviate).

5. (Optional) Activate the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using ``uv``, there is in almost all cases no longer a need to manually activate the virtual environment.

``uv`` will find the ``.venv`` virtual environment in the working directory or any parent directory, and activate it on the fly whenever you run a command via `uv` inside your project folder structure:

.. code:: sh

   uv run <command>

However, you still *can* manually activate the virtual environment if needed.
When developing in an IDE, for instance, this can in some cases be necessary depending on your IDE settings.
To manually activate the virtual environment, run one of the "known" legacy commands:

..on Windows:

.. code:: sh

   .venv\Scripts\activate.bat

..on Linux:

.. code:: sh

   source .venv/bin/activate

6. Install pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``.pre-commit-config.yaml`` file in the project root directory contains a configuration for pre-commit hooks.
To install the pre-commit hooks defined therein in your local git repository, run:

.. code:: sh

   uv run pre-commit install

All pre-commit hooks configured in ``.pre-commit-config.yam`` will now run each time you commit changes.

pre-commit can also manually be invoked, at anytime, using:

.. code:: sh

   uv run pre-commit run --all-files

To skip the pre-commit validation on commits (e.g. when intentionally
committing broken code), run:

.. code:: sh

   uv run git commit -m <MSG> --no-verify

To update the hooks configured in ``.pre-commit-config.yaml`` to their
newest versions, run:

.. code:: sh

   uv run pre-commit autoupdate

7. Test that the installation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To test that the installation works, run pytest in the project root folder:

.. code:: sh

   uv run pytest

Meta
----
Copyright (c) 2026 `DNV <https://www.dnv.com/>`_ AS. All rights reserved.

Siegfried Eisinger - `@LinkedIn <https://www.linkedin.com/in/siegfried-eisinger-a337638b>`_ - siegfried.eisinger@dnv.com

Aleksandar Babic - `@LinkedIn <https://www.linkedin.com/in/aleksandar-babic-no>`_ - aleksandar.babic@dnv.com

Distributed under the MIT license. See `LICENSE <LICENSE.md/>`_ for more information.

`https://github.com/dnv-opensource/crane-controller <https://github.com/dnv-opensource/crane-controller/>`_

Contributing
------------

1. Fork it `<https://github.com/dnv-opensource/crane-controller/fork/>`_
2. Create an issue in your GitHub repo
3. Create your branch based on the issue number and type (``git checkout -b issue-name``)
4. Evaluate and stage the changes you want to commit (``git add -i``)
5. Commit your changes (``git commit -am 'place a descriptive commit message here'``)
6. Push to the branch (``git push origin issue-name``)
7. Create a new Pull Request in GitHub

For your contribution, please make sure you follow the `STYLEGUIDE <STYLEGUIDE.md/>`_ before creating the Pull Request.

.. |pypi| image:: https://img.shields.io/pypi/v/crane-controller.svg?color=blue
   :target: https://pypi.python.org/pypi/crane-controller
.. |versions| image:: https://img.shields.io/pypi/pyversions/crane-controller.svg?color=blue
   :target: https://pypi.python.org/pypi/crane-controller
.. |license| image:: https://img.shields.io/pypi/l/crane-controller.svg
   :target: https://github.com/dnv-opensource/crane-controller/blob/main/LICENSE
.. |ci| image:: https://img.shields.io/github/actions/workflow/status/dnv-opensource/crane-controller/.github%2Fworkflows%2Fnightly_build.yml?label=ci
.. |docs| image:: https://img.shields.io/github/actions/workflow/status/dnv-opensource/crane-controller/.github%2Fworkflows%2Fpush_to_release.yml?label=docs
   :target: https://dnv-opensource.github.io/crane-controller/README.html
