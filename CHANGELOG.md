# Changelog

All notable changes to the [crane-controller] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
* `ProximalPolicyOptimizationAgent.resume()` classmethod to continue training from a saved checkpoint.
  Restores VecNormalize statistics and keeps normalization in training mode, consistent with SB3's
  `PPO.load()` + `.learn(reset_num_timesteps=False)` pattern.
* `reset_num_timesteps` keyword argument to `do_training()` -- pass `False` when resuming to
  preserve the learning rate schedule across training sessions.
* `--resume-from PATH` CLI flag to `scripts/train_ppo.py` for checkpoint-based continued training.
* `ProximalPolicyOptimizationAgent._save_reward_plot()` saves a scatter plot of episode rewards
  vs training step as a PNG alongside the model after each training run.

### Changed
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
