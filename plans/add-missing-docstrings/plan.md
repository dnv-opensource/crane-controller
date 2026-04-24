# Plan: Add missing docstrings

## Baseline (before implementation)

- `ruff check src/ scripts/ --select D100,D101,D102,D103,D104,D105,D107`
  reported many violations across the codebase, including missing module,
  class, method, and function docstrings.
- In `ruff.toml`, all `D10x` rules were temporarily ignored (listed in the
  `ignore` section).
- All tests passed.

## Approach

Work "inside-out" — from the most local scope (magic methods, `__init__`) to
the most global scope (modules, packages). Re-activate `D10x` rules in `ruff.toml`
one at a time as each category is complete.

### Phase 1 — `__init__` docstrings (D107)

Add docstrings to all `__init__` methods. For classes where the class docstring
already documents constructor parameters, `__init__` may have a brief one-line
summary only.

Re-activate: uncomment `D107` from the `ignore` list in `ruff.toml`.

### Phase 2 — Magic method docstrings (D105)

Add docstrings to magic methods (`__repr__`, `__str__`, `__len__`, etc.).

Re-activate: uncomment `D105`.

### Phase 3 — Public function docstrings (D103)

Add docstrings to all public module-level functions and script entry points.

Re-activate: uncomment `D103`.

### Phase 4 — Public method docstrings (D102)

Add docstrings to all public instance and class methods.

Re-activate: uncomment `D102`.

### Phase 5 — Public class docstrings (D101)

Add docstrings to all public classes.

Re-activate: uncomment `D101`.

### Phase 6 — Public module docstrings (D100)

Add module-level docstrings to all `src/**/*.py` and `scripts/*.py` modules.

Re-activate: uncomment `D100`.

### Phase 7 — Public package docstrings (D104)

Add docstrings to all `__init__.py` files in sub-packages.

Re-activate: uncomment `D104`.

## Constraints

- All docstrings must be numpy-style compliant.
- Do not use ambiguous Unicode characters (em-dash `—`, en-dash `–`).
  Use ASCII hyphen `-` instead. Exemptions are characters listed in
  `allowed-confusables` in `ruff.toml`.

## Validation

- `ruff check src/ scripts/ --select D100,D101,D102,D103,D104,D105,D107`
  → **0 violations** ✅
- `ruff check src/ scripts/ tests/` → 0 errors ✅
- `ruff format --check` → no format violations ✅
- `pytest -m "not slow"` → all tests pass ✅

## Definition of Done

- [x] In `ruff.toml`, all temporarily ignored `D10x` rules are activated
  (commented out of the `ignore` section)
- [x] All missing docstrings are added
- [x] `ruff format --check` passes
- [x] `ruff check` passes (0 errors)
- [x] `pytest` — all tests pass
