# Plan: Add type annotations

## 0. Instruction Sources and Precedence

This plan follows `.github/copilot-instructions.md`:

- Primary instruction sources loaded: `/.instructions.md`, `/.prompt.md`, and `plans/add-type-hints/plan-draft.md`.
- Scoped `.github/instructions/*.instructions.md` were checked; none are specific to this task.
- Conflict handling uses the repository precedence model (scoped → `/.instructions.md` → `/.prompt.md`).
- Plan-first workflow is applied for this non-trivial maintenance task.

## 1. Restated Goal

Add missing type annotations across the codebase so that:

- Ruff ANN violations remain at zero.
- Pyright errors drop from 54 to below 10 (target: 0).
- Pyright warnings are reduced significantly.
- All existing tests continue to pass.

The project instructions (`/.instructions.md`) explicitly state:
> *Type annotations are considered an important part of code quality.*
> *Avoid using `Any` unless absolutely necessary.*

## 2. Current Baseline

| Metric | Value |
|---|---|
| `ruff check --select ANN` | 0 violations |
| `ruff check` | 0 violations |
| `ruff format --check` | clean |
| `pyright` errors | **54** |
| `pyright` warnings | **188** |
| `pytest` | all pass |

### Pyright errors by file (top hotspots)

| File | Errors |
|---|---|
| `envs/controlled_crane_pendulum.py` | 27 |
| `q_agent.py` | 8 |
| `algorithm.py` | 3 |
| `tests/test_environment.py` | 3 |
| `envs/controlled_mobile_crane.py` | 2 |
| `reinforce_agent.py` | 2 |
| `tests/test_algorithm.py` | 2 |
| `tests/test_q.py` | 2 |
| `ppo_agent.py` | 1 |
| wrappers (4 files) | 4 |

### Pyright error categories

| Category | Count | Root Cause |
|---|---|---|
| `reportMissingTypeArgument` | 14 | Bare `gym.Env`, `Callable`, `list`, `tuple`, wrapper base classes |
| "Cannot access attribute for class `object`" | ~22 | `crane` parameter typed as `Callable[[], object]` instead of `Callable[[], Crane]` |
| "Argument of type `object` cannot be assigned" | ~10 | `env.reset()` / `env.step()` return `object` due to unparameterized `gym.Env` |
| `MappingProxyType` not assignable to `dict[str, Any]` | 2 | `metadata` override incompatible with base class |
| Tensor / ndarray mismatch | 2 | `reinforce_agent.py` `torch.from_numpy` return |
| `int64` not assignable to `int` | 1 | `env.step(np.random.integers(...))` in test |

## 3. Scope and Constraints

- **In scope:** Python source (`src/**`), tests (`tests/**`), scripts (`scripts/**`).
- **Out of scope:** Unrelated feature work, large architectural refactors not driven by type-correctness.
- Do not add broad rule suppressions in `ruff.toml` or `pyproject.toml`.
- `# type: ignore[code]` only as a last resort, with specific rule code and a justification comment.
- Avoid `Any` except for `*args`/`**kwargs`.
- Maintain deterministic test behavior.

## 4. Working Assumptions

- Toolchain commands available through `uv`.
- `py_crane` ships `py.typed` and exposes typed `Crane`, `Boom`, etc.
- `build_crane() -> Crane` is the canonical factory; environments receive `Callable[[], Crane]` not `Callable[[], object]`.
- `gym.Env` should be parameterized as `gym.Env[ObsType, ActType]` where possible.
- Fixing the `crane` parameter type will resolve the majority of pyright errors (22 of 54).

## 5. Execution Plan

### Phase A: Parameterize generic base classes

**Targets:** `gym.Env`, `gym.RewardWrapper`, `gym.ActionWrapper`, `gym.ObservationWrapper`, `gym.Wrapper`, `Callable`, `list`, `tuple` — wherever pyright reports `reportMissingTypeArgument`.

1. `controlled_crane_pendulum.py`: parameterize `gym.Env[tuple[int, ...] | np.ndarray, int]` (or appropriate obs/act types).
2. `controlled_mobile_crane.py`: parameterize `gym.Env[dict[str, npt.NDArray[np.int_]], int]`.
3. Wrapper modules (`clip_reward.py`, `discrete_actions.py`, `reacher_weighted_reward.py`, `relative_position.py`): add type arguments to wrapper base classes.
4. Tests and scripts: parameterize bare `Callable` to `Callable[..., Crane]` or the specific signature.

**Validation gate:** `uv run pyright src/ tests/ scripts/`, `uv run ruff check`, `uv run pytest`.

### Phase B: Fix `crane` parameter type (`object` → `Crane`)

**Root cause:** `controlled_crane_pendulum.py.__init__` currently types `crane` as `Callable[[], object]`. Pyright correctly reports that `.position`, `.velocity`, `.boom_by_name`, etc. are inaccessible on `object`.

1. Change `crane: Callable[[], object]` → `crane: Callable[[], Crane]` in `AntiPendulumEnv.__init__`, importing `Crane` from `py_crane.crane`.
2. Type `self.crane` as `Crane` and `self.wire` as the appropriate boom type.
3. Propagate to `controlled_mobile_crane.py` if needed (constructor already takes `crane: Crane`).
4. Verify that `build_crane` signature matches `Callable[..., Crane]`.

**Validation gate:** same as Phase A.

### Phase C: Fix `env.reset()` / `env.step()` return type narrowing

**Root cause:** When `gym.Env` is not parameterized or the obs type is a union, pyright infers `object` for obs returned from `reset()`/`step()`. Agent methods that accept `tuple[int, int, int, int, int]` then reject the `object`.

1. Ensure environment `reset()` and `step()` have explicit return type annotations that match the parameterized `Env`.
2. In agent modules (`algorithm.py`, `q_agent.py`, `ppo_agent.py`), adjust the `env` attribute type to match the parameterized environment, or use explicit casts/assertions at the boundary.
3. Address the `MappingProxyType` vs `dict[str, Any]` conflict for `metadata` — either switch back to a plain dict or use `ClassVar` with the correct type and a targeted `type: ignore[override]`.

**Validation gate:** same as Phase A.

### Phase D: Fix remaining edge-case errors

1. `reinforce_agent.py`: resolve `Tensor` ↔ `ndarray` mismatch in `sample_action` (`torch.from_numpy` returns `Tensor`, not `ndarray`).
2. `tests/test_environment.py`: cast `np.int64` from `np_random.integers()` to `int` before calling `env.step()`.
3. `q_agent.py`: add type argument to `defaultdict` and bare `list` in `dump_results`.
4. Add any remaining missing generic type arguments flagged by pyright.

**Validation gate:** same as Phase A.

### Phase E: Reduce pyright warnings

After errors reach zero, scan the warning report (188 warnings) and reduce the most impactful categories:

1. `reportUnknownParameterType` — tighten `Callable` type args.
2. `reportUnknownMemberType` — narrow attribute types where feasible.
3. Skip warnings that originate in third-party stubs (matplotlib, pandas) as those are not actionable.

**Exit criteria:** warnings reduced meaningfully; no new errors introduced.

### Phase F: Final validation and formatting

1. `uv run ruff format`
2. `uv run ruff check`
3. `uv run ruff check --select ANN`
4. `uv run pyright src/ tests/ scripts/`
5. `uv run pytest`
6. Update `CHANGELOG.md` — append to `Unreleased` section.

## 6. Change Management and Safety

- Incremental commits by phase to simplify review and rollback.
- Avoid touching unrelated files.
- Prefer explicit, readable annotations over clever compact rewrites.
- Run the full validation gate after each phase before proceeding.

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `gym.Env` generic parameterization may conflict with `stable_baselines3` expectations | Test `ppo_agent` specifically; use `TYPE_CHECKING` guarded overloads if needed |
| `MappingProxyType` metadata may break at runtime if Gymnasium checks `isinstance(metadata, dict)` | Verify with `pytest`; revert to `dict` if runtime failures occur |
| `py_crane` types may be incomplete despite `py.typed` | Inspect at runtime; add a local stub file under `stubs/` if needed |

## 8. Definition of Done

- [ ] `uv run ruff check --select ANN` — 0 violations
- [ ] `uv run ruff check` — 0 violations (no regression)
- [ ] `uv run ruff format --check` — clean
- [ ] `uv run pyright src/ tests/ scripts/` — fewer than 10 errors (target: 0)
- [ ] `uv run pytest` — all tests pass
- [ ] No rules added to ignore lists in `ruff.toml` / `pyproject.toml` without documented justification
- [ ] `CHANGELOG.md` has an `Unreleased` entry for the type-annotation changes
