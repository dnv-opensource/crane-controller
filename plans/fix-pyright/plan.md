# Plan: Fix issues raised by pyright

## Problem Statement

`pyright` currently raises 2 errors and 2 warnings across the codebase.
The goal is to reduce the error count to **zero** and eliminate warnings where practical — by fixing the underlying code, not by suppressing rules.

Both errors and both warnings are in `tests/test_environment.py` (lines 116–117), in the `test_observations_are_float` test function.

## Baseline

| Metric | Value |
|--------|-------|
| `uv run pyright` errors | 2 |
| `uv run pyright` warnings | 2 |
| `uv run pyright` informations | 0 |

### Error details

All issues originate from `tests/test_environment.py`, function `test_observations_are_float`:

| Line | Code | Rule | Description |
|------|------|------|-------------|
| 116 | `obs.dtype` | `reportAttributeAccessIssue` | Cannot access attribute `dtype` for class `tuple[int, ...]` |
| 116 | `obs.dtype` | `reportUnknownMemberType` | Type of `dtype` is partially unknown |
| 117 | `obs.astype(int)` | `reportAttributeAccessIssue` | Cannot access attribute `astype` for class `tuple[int, ...]` |
| 117 | `obs.astype(int)` | `reportUnknownMemberType` | Type of `astype` is partially unknown |

### Root cause

`AntiPendulumEnv` is generic over `AntiPendulumObs = tuple[int, ...] | np.ndarray`.
The `step()` method returns `tuple[AntiPendulumObs, float, bool, bool, dict[...]]`.
When pyright unpacks the return value, `obs` receives the union type `tuple[int, ...] | np.ndarray`.
The test `test_observations_are_float` creates an env in **continuous mode** (no `discrete` argument), so at runtime `obs` is always an `np.ndarray` — but pyright cannot narrow this statically.

The `.dtype` and `.astype()` attributes exist on `np.ndarray` but not on `tuple[int, ...]`, so pyright raises `reportAttributeAccessIssue` errors for the `tuple[int, ...]` branch of the union.
The `reportUnknownMemberType` warnings are cascading from the same union ambiguity.

## Context

- **Relevant modules:**
  - `tests/test_environment.py` — contains the 2 errors
  - `src/crane_controller/envs/controlled_crane_pendulum.py` — defines `AntiPendulumObs`, `AntiPendulumEnv`, `step()`, `reset()`
- **Related tests:** `tests/test_environment.py` (must continue to pass)
- **Existing suppressions:** 21 `# type: ignore` comments across `src/` and `tests/` (pre-existing; not in scope unless they become unnecessary)
- **Constraints:**
  - Do **not** suppress rules globally in `pyproject.toml`
  - Use per-line `# pyright: ignore` only as a last resort
  - Run `uv run ruff format` after changes
  - Existing tests must continue to pass

## Assumptions

- The test `test_observations_are_float` is correct and should continue to test ndarray behavior.
- The `AntiPendulumObs` union type is intentional and should be preserved (discrete vs. continuous mode).
- Fixing the test-side type narrowing is preferable to changing the library's public API types.

## Proposed Approach

### Phase 1: Fix the 2 errors (and 2 cascading warnings) in `test_environment.py`

Add a type-narrowing assertion (`assert isinstance(obs, np.ndarray)`) after the `env.step()` call in `test_observations_are_float`. This:
- Documents the test's expectation that continuous-mode observations are ndarrays
- Lets pyright narrow the union to `np.ndarray`, eliminating both `reportAttributeAccessIssue` errors
- Eliminates the cascading `reportUnknownMemberType` warnings (since `np.ndarray.dtype` and `.astype()` are fully typed)
- Doubles as a runtime assertion (the test would fail with a clear message if the contract changes)

Concrete change in `tests/test_environment.py`, function `test_observations_are_float`:

```python
def test_observations_are_float(crane: Callable[..., Crane]) -> None:
    """Test that observations preserve sub-integer precision after a physics step."""
    env = AntiPendulumEnv(crane)
    _ = env.reset()
    obs, _, _, _, _ = env.step(1)  # one physics step produces fractional values
    assert isinstance(obs, np.ndarray)       # <-- NEW: narrows type for pyright
    assert obs.dtype == np.float64
    assert not np.all(obs == obs.astype(int))  # sub-integer precision is preserved
```

### Phase 2: Cleanup — check for stale `# type: ignore` / `# pyright: ignore` comments

After the fix, re-run `uv run pyright` and confirm:
- 0 errors, 0 warnings, 0 `reportUnnecessaryTypeIgnoreComment` informations
- No stale suppression comments were introduced or left behind

### Phase 3: Validation

1. `uv run pyright` — 0 errors
2. `uv run pytest` — all tests pass
3. `uv run ruff format --check` — no format violations
4. `uv run ruff check src/ tests/ scripts/` — no new lint issues

## Alternatives Considered

- **Option A: Cast `obs` to `np.ndarray`** — e.g. `obs = np.asarray(obs)`. This changes runtime semantics (creates a copy/view) and obscures the test's intent. Rejected.
- **Option B: Add a `# pyright: ignore` comment** — suppresses the symptom without documenting the invariant. Against the plan-draft constraint of avoiding suppressions. Rejected.
- **Option C: Overload `step()` based on discrete/continuous mode** — would require significant refactoring of the `AntiPendulumEnv` class (e.g. generic type parameter for mode). Disproportionate effort for 2 errors in a test file. Could be a future improvement but out of scope here. Rejected for now.
- **Option D: `assert isinstance` (chosen)** — idiomatic Python type-narrowing, zero runtime cost in practice, documents the test contract, and fixes all 4 diagnostics. Preferred.

## Risks & Mitigations

- **Risk:** The `isinstance` assertion could fail if the env's `step()` ever returns a tuple in continuous mode.
  - **Mitigation:** This would be a genuine regression — the assertion catches it with a clear error message. This is desirable.
- **Risk:** Future pyright versions change union narrowing behavior.
  - **Mitigation:** `isinstance` narrowing is a well-established PEP 647 pattern; unlikely to break.

## Definition of Done

- [x] Code implemented (isinstance assertion added)
- [x] `uv run pyright` — 0 errors, 0 warnings, 0 informations
- [x] `uv run pytest` — all 11 tests pass
- [x] `uv run ruff format --check` — 46 files already formatted
- [x] `uv run ruff check src/ tests/ scripts/` — all checks passed
- [x] No rules added to the `ignore` list in `pyproject.toml`
