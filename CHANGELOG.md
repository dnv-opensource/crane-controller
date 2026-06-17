# Changelog

All notable changes to the [crane-controller] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
* Five new `RewardConfig` fields for a principled derivatives-based reward design:
  `angle` (-theta^2), `angular_velocity` (-theta_dot^2), `crane_velocity` (-x_dot^2),
  `crane_acceleration` (-x_ddot^2), `angular_acceleration` (-theta_ddot^2). All default to 0.0
  for full backward compatibility with existing configs using `energy`.
  Angular velocity uses pure theta_dot = `(cm_v[0] - origin_v[0]) / wire.length`,
  excluding crane translation. Angular acceleration is computed via one-step
  finite difference of theta_dot; zero on the first step after each episode reset.
* `AntiPendulumEnv` continuous observation `obs[3]` changed from absolute load
  x-velocity (`wire.cm_v[0]`) to pure angular velocity theta_dot (rad/s), making the
  observation independent of crane translation velocity.
* `experiments/derivatives_baseline.yaml`: starting config for the derivatives reward.
* `experiments/hybrid_cv01.yaml`: validated hybrid config (energy + crane_velocity + position
  return). Seeds 2718, 3141, 31415 achieve 6/6 OOD generalisation at start_speed=7.0.
* `start_speed` field in `TrainingConfig` (default 1.0); wired through `train_ppo.py`
  (`--start-speed`) and `play_ppo.py` (`--start-speed`). With `randomize_start=True` acts
  as the upper bound of the per-episode speed sampling range `+-[min_speed, start_speed]`.
* `randomize_start` field in `TrainingConfig`; wired through both scripts.
  `play_ppo.py` pre-parses `--model-path` to auto-load `randomize_start` from the model
  sidecar; `--randomize-start` / `--no-randomize-start` override it.
* `RewardConfig`, `TrainingConfig`, and `ExperimentConfig` frozen dataclasses in new module
  `src/crane_controller/experiment_config.py`. Replace the opaque `reward_fac` tuple with
  named fields, eliminating the silent index-swap bug class.
* YAML experiment config support in `train_ppo.py` via `--config PATH`.
  Missing YAML keys fall back to `RewardConfig`/`TrainingConfig` defaults.
* `--reward-fac ENERGY POSITIONAL TIME POSITION ACCELERATION` CLI override on `train_ppo.py`;
  takes precedence over `--config`.
* JSON sidecar (`*_meta.json`) written alongside every saved model by `train_ppo.py` and read
  automatically by `play_ppo.py` — reward weights follow the model without manual flags.
* `terminal_penalty` field in `RewardConfig`: one-time reward added on episode truncation
  (OOB crash). Defaults to 0.0 (disabled). Used in `hybrid_cv01.yaml` as -5.0.
* `seed`, `ent_coef`, `learning_rate`, `clip_range`, `n_steps` parameters on
  `ProximalPolicyOptimizationAgent.__init__` and corresponding CLI flags in `train_ppo.py`.
* `gamma` parameter on `ProximalPolicyOptimizationAgent` (default 0.99) and `--gamma` CLI flag in
  `train_ppo.py` to configure the PPO discount factor without editing source code.
* `continuous_actions: bool` parameter on `AntiPendulumEnv` (default `False`). When `True`, the
  action space is `Box([-1], [1])` and the action value is scaled by `acc` to produce crane
  acceleration, enabling PPO to produce any acceleration in `[-acc, +acc]`. When `False` (default),
  the action space remains `Discrete(3)` for full Q-agent backward compatibility.
  `TrainingConfig.continuous_actions` (default `True`) and `--continuous-actions` /
  `--no-continuous-actions` CLI flags in both `train_ppo.py` and `play_ppo.py` control this for
  PPO workflows; Q-agent workflows pass `continuous_actions=False` explicitly.
  `ppo_agent.do_one_episode()` updated to pass actions without casting to `int`, so both action
  space types work correctly during inference.

### Fixed
* Fixed general incompatibilities between the updated repository and changes in `eis`branch.
* `ProximalPolicyOptimizationAgent.load()` now applies a `TimeLimit` wrapper (max 3000 steps),
  matching the training configuration. Without it, `play_ppo.py` ran indefinitely on a converged
  model whose near-zero reward never crossed the termination threshold.
* `AntiPendulumEnv.render()` now handles `render_mode='plot'` by calling `show_plot()` directly,
  so the episode plot appears when running `play_ppo.py --render-mode plot`.
* `show_plot()` legend now includes lines from twin y-axes (load speed, crane speed, damping) by
  combining handles from both axes with `get_legend_handles_labels()`.
* `show_plot()` title moved from `plt.title()` (attached to last axes) to `plt.suptitle()`
  (figure-level), preventing the title from appearing between subplots.
* `show_plot()` switched from 2×2 grid to 4×1 vertical layout (16×12 in) so all subplots share
  a common time axis and each has full width.
* Disabled explicit time penalty (`reward_fac[2] = 0.0`) in PPO training and playback scripts.
  The term `−self.time × 0.001` uses hidden state absent from the observation, violating the
  Markov property and destabilising PPO's value function. Time preference is already encoded
  implicitly through the discount factor γ.

* `ProximalPolicyOptimizationAgent.resume()` classmethod to continue training from a saved checkpoint.
  Restores VecNormalize statistics and keeps normalization in training mode, consistent with SB3's
  `PPO.load()` + `.learn(reset_num_timesteps=False)` pattern.
* `reset_num_timesteps` keyword argument to `do_training()` -- pass `False` when resuming to
  preserve the learning rate schedule across training sessions.
* `--resume-from PATH` CLI flag to `scripts/train_ppo.py` for checkpoint-based continued training.
* `ProximalPolicyOptimizationAgent._save_reward_plot()` saves a scatter plot of episode rewards
  vs training step as a PNG alongside the model after each training run.

### Changed
* Set many tests to 'skip'. These need to be updated (or deleted if not relevant any more) as soon as possible
* configuration of environments and agents moved to dataclass objects, to avoid lengthy argument lists. Related changes.
* Removed unused reward factors in reward calculation
* Adapted the y-size of plots, such that it fits also smaler screens without scrolling
* `AntiPendulumEnv` parameter `size` renamed to `rail_limit`; `TrainingConfig.size` renamed to
  `rail_limit`; `--size` CLI flag renamed to `--rail-limit`. Semantics unchanged: half-span of
  the crane rail in metres (crane spans +-rail_limit).
* `show_plot()` rewritten with 6 individual subplots (load angle, load speed + damping curve,
  crane position + origin line, crane speed, rewards, x-acceleration), replacing the previous
  `twinx()`-based layout that caused overlapping scales and colliding legends.
* Moved `logging.basicConfig` to the top of `main()` in `train_ppo.py` and `play_ppo.py` so
  logging is configured before any application logic runs.
* Refactored `ProximalPolicyOptimizationAgent` API to separate training and inference concerns:
  * Constructor (`__init__`) is now training-only; accepts `save_path: str | None` instead of `trained: tuple`.
  * New `load()` classmethod loads a saved model for inference, mirroring the SB3 `PPO.load()` convention.
  * Removed the `n_envs=0` magic value that previously signalled inference mode.
* Added `test_ppo_inference_disables_training_mode` test covering the `load()` path.
* Updated `README.rst` test file list to reflect actual test module names.

### Removed
* Removed `reinforce_agent.py` (early prototype superseded by PPO) and dropped `torch` as a direct dependency (still available transitively via `stable-baselines3`). Also removed the associated `torch` type stubs and the CUDA optional dependency.

### Changed
* CI: Skip slow `test_algorithm_strategies` test in GitHub workflow runs by adding a `slow` pytest marker and passing `-m "not slow"` in `_test.yml` and `_test_future.yml`
* Adjusted and partly amended package structure to be in sync with latest changes in python_project_template v0.2.11
* Typing:
  * Added type annotations across all source, test, and script modules
  * Added type stubs for `torch`, `matplotlib`, and `stable-baselines3`
* Docstrings:
  * Reformatted all existing docstrings to numpy-style
  * Added missing docstrings across all source and script modules
* Resolved all issues raised by `ruff`, `pyright`, and `mypy`

### Fixed
* Restored deterministic AntiPendulum environment seeding and aligned reset/step behavior with the existing environment tests.


## [0.0.2] - YYYY-MM-DD

### Changed
* ...


## [0.0.1] - YYYY-MM-DD

* Initial release

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
  * function x in module z

### Removed

* following features have been removed:
  * function y in module z


<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-opensource/crane-controller/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/dnv-opensource/crane-controller/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/dnv-opensource/crane-controller/releases/tag/v0.0.1
[crane-controller]: https://github.com/dnv-opensource/crane-controller
