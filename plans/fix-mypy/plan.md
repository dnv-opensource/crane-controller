# Plan: Fix issues raised by mypy

## Problem Statement

`mypy` currently raises 1 error (without `--warn-unused-ignores`) or 14 errors (with `--warn-unused-ignores`: 1 real + 13 stale suppression comments).
The goal is to reach **0 errors** with both `uv run mypy` and `uv run mypy --warn-unused-ignores`, by fixing the underlying code — not by adding new suppressions.

## Baseline

| Metric | Value |
|--------|-------|
| `uv run mypy` errors | 1 |
| `uv run mypy --warn-unused-ignores` errors | 14 (1 real + 13 unused-ignore) |
| `uv run pyright` | 0 errors, 0 warnings, 0 informations |
| `uv run pytest` | 11 passed |
| `uv run ruff format --check` | clean |

### Error details

#### Real error (1)

| File | Line | Code | Description |
|------|------|------|-------------|
| `tests/test_ppo.py` | 49 | `attr-defined` | `"dict[str \| Any, RunningMeanStd]"` has no attribute `"mean"` |

**Root cause:** `stable_baselines3` ships a `py.typed` marker, so mypy uses the library's own source types instead of our stubs. In `VecNormalize.__init__`, `obs_rms` is assigned as a `dict[str, RunningMeanStd]` in the Dict-observation branch, and the non-Dict branch `RunningMeanStd(...)` is annotated with `# type: ignore[assignment]` in the library code — so mypy only sees the dict type. The test accesses `agent.vec_env.obs_rms.mean`, which is valid for `RunningMeanStd` but not for `dict`.

#### Stale `# type: ignore` comments (13)

**Group A — entire comment unused (10 lines):**

| File | Line | Suppressed code | Justification comment |
|------|------|-----------------|-----------------------|
| `controlled_crane_pendulum.py` | 92 | `[assignment]` | Gymnasium metadata typing is loose |
| `controlled_crane_pendulum.py` | 137 | `[type-arg]` | Discrete type arg not needed here |
| `controlled_crane_pendulum.py` | 277 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 307 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 327 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 351 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 428 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 435 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_crane_pendulum.py` | 437 | `[attr-defined]` | dynamic attr on Wire |
| `controlled_mobile_crane.py` | 54 | `[assignment]` | *(none)* |

**Group B — specific code unused within multi-code comment (3 lines):**

| File | Line | Full comment | Unused code | Still needed |
|------|------|--------------|-------------|--------------|
| `controlled_crane_pendulum.py` | 214 | `[arg-type,list-item]` | `arg-type` | `list-item` |
| `controlled_crane_pendulum.py` | 218 | `[arg-type,list-item]` | `list-item` | `arg-type` |
| `controlled_crane_pendulum.py` | 229 | `[arg-type,call-overload]` | `arg-type` | `call-overload` |

### Important constraint: pyright / mypy divergence

Pyright currently reports **0 informations** for `reportUnnecessaryTypeIgnoreComment`, meaning pyright still needs some or all of the `# type: ignore` comments that mypy considers stale. Both tools recognize `# type: ignore` as a suppression directive.

- `# type: ignore` — suppresses diagnostics for **both** mypy and pyright.
- `# pyright: ignore` — suppresses diagnostics for **pyright only**.

Removing a `# type: ignore` that mypy no longer needs but pyright still needs requires replacing it with `# pyright: ignore[rule]`.

## Context

- **Relevant files:**
  - `tests/test_ppo.py` — 1 real error (line 49)
  - `src/crane_controller/envs/controlled_crane_pendulum.py` — 12 stale ignores
  - `src/crane_controller/envs/controlled_mobile_crane.py` — 1 stale ignore
  - `stubs/stable_baselines3-stubs/common/vec_env.pyi` — `RunningMeanStd` stub (mypy ignores it due to `py.typed`)
- **Related tests:** all (`uv run pytest`)
- **Constraints:**
  - Do **not** suppress rules globally in `pyproject.toml`
  - Suppress per-line with `# type: ignore` only as a last resort
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass
  - Pyright must remain at 0 errors / 0 warnings after changes

## Assumptions

- `stable_baselines3` will continue to ship `py.typed`, so stubs do not affect mypy.
- Removing stale `# type: ignore` comments that pyright still needs requires converting them to `# pyright: ignore[rule]`.
- The `AntiPendulumEnv` uses non-Dict observation spaces in all current tests, so `obs_rms` is always a `RunningMeanStd` at runtime.

## Proposed Approach

### Phase 1: Fix the real mypy error in `tests/test_ppo.py`

Add a type-narrowing assertion before the `.mean` access:

```python
from stable_baselines3.common.vec_env import RunningMeanStd
# ...
assert isinstance(agent.vec_env.obs_rms, RunningMeanStd)
assert not np.allclose(agent.vec_env.obs_rms.mean, 0.0)
```

This:
- Narrows `obs_rms` from `dict[str, RunningMeanStd] | RunningMeanStd` to `RunningMeanStd`
- Documents the test's assumption about observation-space type
- Acts as a runtime guard if the contract changes

**Validate:** `uv run mypy`, `uv run pyright`, `uv run pytest`

### Phase 2: Remove stale `# type: ignore` comments

For each stale comment (Groups A and B):

1. Remove or trim the `# type: ignore` comment.
2. Run `uv run pyright` to check if pyright now raises a diagnostic on that line.
   - **If yes:** add `# pyright: ignore[rule]` with the specific pyright rule name, preserving the original justification comment.
   - **If no:** leave the comment removed (the diagnostic was specific to mypy and is now resolved upstream).
3. For Group B lines (multi-code comments): remove only the unused code(s), keep the still-needed one(s).

Work file-by-file:
- `src/crane_controller/envs/controlled_crane_pendulum.py` (12 comments)
- `src/crane_controller/envs/controlled_mobile_crane.py` (1 comment)

**Validate after each file:** `uv run mypy --warn-unused-ignores`, `uv run pyright`, `uv run ruff format`

### Phase 3: Final validation

1. `uv run mypy` — 0 errors
2. `uv run mypy --warn-unused-ignores` — 0 errors
3. `uv run pyright` — 0 errors, 0 warnings, 0 informations
4. `uv run pytest` — all tests pass
5. `uv run ruff format --check` — clean
6. `uv run ruff check src/ tests/ scripts/` — clean

## Alternatives Considered

- **Option A: Update the stubs to fix the mypy error** — The `stable_baselines3` stubs declare `obs_rms: RunningMeanStd`, but mypy ignores them because the library has `py.typed`. Updating the stubs won't help mypy. Rejected.
- **Option B: Add `# type: ignore[attr-defined]` to test_ppo.py** — Suppresses rather than fixes. Against plan-draft constraints. Rejected.
- **Option C: Remove all `# type: ignore` comments at once and fix the fallout** — Risky; could produce cascading errors from both tools. Prefer incremental approach. Rejected.
- **Option D: Incremental removal with pyright-specific fallback (chosen)** — Safe, testable at each step, preserves pyright compatibility. Preferred.

## Risks & Mitigations

- **Risk:** Removing a `# type: ignore` that pyright needs causes pyright regressions.
  - **Mitigation:** Check pyright after each removal; convert to `# pyright: ignore[rule]` where needed.
- **Risk:** Future `stable_baselines3` updates change `obs_rms` typing.
  - **Mitigation:** The `isinstance` assertion will catch this as a test failure.
- **Risk:** Removing stale ignores in `controlled_crane_pendulum.py` unmasks latent type errors in either tool.
  - **Mitigation:** Run both type checkers after each file. If real errors surface, fix them before proceeding.

## Definition of Done

- [x] Code implemented (all phases complete)
- [x] `uv run mypy` — 0 errors
- [x] `uv run mypy --warn-unused-ignores` — 0 unused-ignore errors
- [x] `uv run pyright` — 0 errors, 0 warnings, 0 informations
- [x] `uv run pytest` — all 11 tests pass
- [x] `uv run ruff format --check` — no format violations
- [x] `uv run ruff check src/ tests/ scripts/` — no new lint issues
- [x] No rules added to the `ignore` list in `pyproject.toml`
