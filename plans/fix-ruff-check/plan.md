# Plan: Fix issues raised by ruff check

## Baseline (before implementation)

- `ruff check src/ scripts/` reported multiple violations across the `src/` tree,
  including unused imports, missing type annotations, logging format issues, and
  various style violations.
- `ruff format --check` reported no format violations.
- All tests passed.

## Approach

Work in phases, ordered from easiest mechanical fixes to the most structural refactors.
Run `ruff format`, `ruff check`, and `pytest` after each phase.

### Phase 1 — Auto-fixable violations

Run `ruff check --fix` to automatically resolve all auto-fixable violations
(unused imports, redundant escape sequences, etc.).

### Phase 2 — Manual fixes per violation category

Address remaining violations manually, grouped by rule category:

- `ANN` — Missing type annotations (deferred to *add-type-hints* plan)
- `D` — Docstring violations (deferred to *reformat-existing-docstrings* and
  *add-missing-docstrings* plans)
- `G` — Logging format violations: replace f-string/format() calls with `%`-style
  logging args.
- `UP` — Pyupgrade / modernization fixes.
- `RUF` — Ruff-specific rules (unnecessary noqa, etc.).
- `FURB` — Modernization rules (single-item membership tests, etc.).
- `PTH` — Use `pathlib` instead of `os.path`.

### Phase 3 — `_ =` assignments for suppressed return values

Add `_ =` assignments where return values of calls are unused (ruff RET rules).

## Validation

- `ruff check src/ scripts/ tests/` → **0 errors** ✅
- `ruff format --check` → no format violations ✅
- `pytest -m "not slow"` → all tests pass ✅

## Definition of Done

- [x] Code implemented (all phases complete)
- [x] `ruff check` reports 0 errors
- [x] `pytest` — all tests pass
- [x] `ruff format --check` — no format violations
- [x] No rules added to the `ignore` list in `ruff.toml` without documented justification
