# Plan: Add type stubs for 3rd party packages

## Baseline (before implementation)

- `pyright src/ scripts/` reported many `reportMissingTypeStubs` and
  `reportUnknownMemberType` errors/warnings for `torch`, `matplotlib`, and
  `stable_baselines3`.
- `mypy src/ scripts/` reported import-untyped errors for the same packages.
- All tests passed.

## Third-party packages requiring stubs

| Package | Usage in codebase | Priority |
|---|---|---|
| `torch` | `reinforce_agent.py` (model, optimizer, tensor ops) | High |
| `matplotlib` | `algorithm.py`, `ppo_agent.py`, `envs/` (plots) | High |
| `stable_baselines3` | `ppo_agent.py` (PPO, make_vec_env, evaluate_policy, VecNormalize) | High |

## Stub structure

Stubs are placed in `stubs/<package-name>-stubs/` and registered in `pyproject.toml`
under `[tool.pyright] stubPath = "stubs"` (or equivalent mypy config).

### `stubs/torch-stubs/`

Cover the torch symbols used in `reinforce_agent.py`:
- `torch.Tensor`, `torch.FloatTensor`, `torch.nn.Module`, `torch.nn.Linear`,
  `torch.nn.Softmax`, `torch.optim.Adam`, `torch.optim.Optimizer`,
  `torch.distributions.Categorical`, `torch.device`,
  `torch.cuda.is_available`, `torch.set_default_device`

### `stubs/matplotlib-stubs/`

Cover matplotlib symbols used across the codebase:
- `matplotlib.pyplot` (`plot`, `show`, `subplots`, `ion`, `tight_layout`, etc.)
- `matplotlib.lines.Line2D`
- `matplotlib.axes.Axes`

### `stubs/stable_baselines3-stubs/`

Cover stable_baselines3 symbols used in `ppo_agent.py`:
- `stable_baselines3.PPO` (constructor, `load`, `learn`, `save`, `predict`)
- `stable_baselines3.common.env_util.make_vec_env`
- `stable_baselines3.common.evaluation.evaluate_policy`
- `stable_baselines3.common.vec_env.VecEnv`
- `stable_baselines3.common.vec_env.VecNormalize` (added post-merge to fix regression)
- `stable_baselines3.common.vec_env.VecEnvWrapper`
- `stable_baselines3.common.vec_env.DummyVecEnv`
- `stable_baselines3.common.running_mean_std.RunningMeanStd`
- `stable_baselines3.common.callbacks.BaseCallback`

## Validation

- `pyright src/ scripts/` → fewer than 5 errors (target: 0) ✅ (after stub regression fix)
- `mypy src/ scripts/` → 0 errors ✅
- `ruff check src/ scripts/ tests/` → 0 errors ✅
- `ruff format --check` → no format violations ✅
- `pytest -m "not slow"` → all tests pass ✅

## Definition of Done

- [x] Type stubs added for `torch`, `matplotlib`, `stable_baselines3`
- [ ] `pyright` reports fewer than 5 errors (target: 0) — see fix-pyright plan for remaining warnings
- [x] `mypy` reports 0 errors
- [x] `ruff format --check` — no format violations
- [x] `ruff check` — no regressions
- [x] `pytest` — all tests pass
- [x] No rules added to ignore lists without documented justification

## Notes on VecNormalize (post-merge regression)

The merge of PR #3 (VecNormalize integration) introduced a regression:
`VecNormalize` was imported in `ppo_agent.py` from
`stable_baselines3.common.vec_env`, but was not covered by the existing stubs.
This caused 1 pyright error and 27 warnings. The fix involves adding
`VecNormalize`, `VecEnvWrapper`, `DummyVecEnv`, and `RunningMeanStd` stubs
to `stubs/stable_baselines3-stubs/common/vec_env.pyi`.
