# Plan: Fix issues raised by mypy

## Baseline (before implementation)

- `mypy src/ scripts/` reported errors, primarily `import-untyped` errors for
  `torch`, `matplotlib`, and `stable_baselines3`.
- After adding type stubs (see *add-type-stubs* plan), the error count dropped to 0.
- All tests passed.

## Approach

Work in phases, ordered from easiest mechanical fixes to the most structural refactors.
Run `ruff format`, `mypy`, and `pytest` after each phase.

### Phase 1 — Add type stubs for untyped packages

See *add-type-stubs* plan. Adding stubs for `torch`, `matplotlib`, and
`stable_baselines3` in the `stubs/` directory, and registering them via
`mypy_path = ["stubs"]` in `pyproject.toml`, resolves all `import-untyped` errors.

### Phase 2 — Fix `attr-defined` errors

Address attribute access errors on typed objects where mypy cannot infer the
attribute.

### Phase 3 — Fix `arg-type` errors

Adjust argument type mismatches flagged by mypy, particularly for:
- NumPy array types (`NDArray` vs `np.ndarray`)
- Optional parameters

### Phase 4 — Fix `assignment` errors

Correct type assignment errors, e.g. assigning a `tuple[str | Path, bool]` to
a `tuple[str | Path, bool] | None` attribute.

### Phase 5 — Fix remaining `type: ignore` comments

Review all `# type: ignore` directives and remove those that are no longer
needed after other fixes.

## Validation

- `mypy src/ scripts/` → **0 errors** ✅
- `pytest -m "not slow"` → all tests pass ✅
- `ruff format --check` → no format violations ✅

## Definition of Done

- [x] Code implemented (all phases complete)
- [x] `mypy` reports 0 errors
- [x] `pytest` — all tests pass
- [x] `ruff format --check` — no format violations
- [x] No rules added to the `ignore` list in `pyproject.toml` without documented justification
