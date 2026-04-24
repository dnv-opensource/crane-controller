# Plan: Add type annotations

## Baseline (before implementation)

- `ruff check src/ scripts/ tests/ --select ANN` reported many missing type
  annotation violations across the codebase.
- `ruff format --check` reported no format violations.
- All tests passed.

## Approach

Work in phases, ordered from easiest-to-resolve types to hardest-to-resolve types.
Run `ruff format`, `ruff check --select ANN`, and `pytest` after each phase.

### Phase 1 — Return types for simple functions/methods

Add `-> None`, `-> str`, `-> int`, `-> float`, `-> bool` return types to functions
and methods that have trivially inferable return types.

### Phase 2 — Parameter types for public functions/methods

Add type annotations to all public function and method parameters using:
- Built-in types: `int`, `float`, `str`, `bool`
- Standard library: `pathlib.Path`, `collections.abc.Callable`, `collections.abc.Sequence`
- `typing` module: `Any`, `Literal`, `TYPE_CHECKING`
- NumPy: `np.ndarray`, `NDArray[np.floating[Any]]`
- Package-specific: Gymnasium `Env`, `spaces.*`

### Phase 3 — Complex types and generics

Add type annotations for:
- Class attributes (`training_error: list[float]`, etc.)
- Generic container types
- Union types (`str | Path`, `int | None`, etc.)
- `tuple[type, ...]` annotations

### Phase 4 — `from __future__ import annotations`

Add `from __future__ import annotations` at the top of each module to enable
deferred evaluation of annotations, allowing forward references.

### Phase 5 — TYPE_CHECKING guard for circular imports

Move type-only imports under `if TYPE_CHECKING:` to avoid circular imports and
keep runtime overhead minimal.

## Validation

- `ruff check src/ scripts/ tests/ --select ANN` → **0 violations** ✅
- `ruff check src/ scripts/ tests/` → 0 errors ✅
- `ruff format --check` → no format violations ✅
- `pytest -m "not slow"` → all tests pass ✅

## Definition of Done

- [x] Type annotations added (all phases complete)
- [x] `ruff --select ANN` reports 0 missing type annotations
- [x] `ruff format --check` — no format violations
- [x] `ruff check` — no regressions
- [x] `pytest` — all tests pass
- [x] No rules added to the `ignore` lists without documented justification
