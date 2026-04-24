# Plan: Reformat existing docstrings

## Baseline (before implementation)

- `ruff check src/ scripts/ --select D` reported various docstring-style violations,
  including non-numpy-style sections, missing `Parameters`/`Returns`/`Raises`
  sections, and wrong blank-line conventions.
- `ruff format --check` reported no format violations.
- All tests passed.

## Approach

Work file by file, reformatting existing docstrings to numpy-style.

### Phase 1 — Module docstrings

Reformat module-level docstrings in all `src/**/*.py` and `scripts/*.py` files to
follow numpy-style (summary line, optional description, Examples section using
`.. code-block:: bash` for CLI usage).

### Phase 2 — Class docstrings

Reformat class docstrings:
- Summary line.
- Extended description (if needed).
- `Parameters` section listing constructor parameters and their types/defaults.

### Phase 3 — Method and function docstrings

Reformat method/function docstrings:
- Summary line.
- `Parameters` section (complete, with types and defaults).
- `Returns` section (for non-None return values, with type and description).
- `Raises` section (where exceptions are raised).

### Phase 4 — Lint and format pass

Run `ruff format` and `ruff check --select D` to verify no violations remain.

## Constraints

- Do not use ambiguous Unicode characters (em-dash `—`, en-dash `–`).
  Use ASCII hyphen `-` instead.
- Characters listed in `allowed-confusables` in `ruff.toml` are exempt.

## Validation

- `ruff check src/ scripts/ --select D` → **0 violations** ✅
- `ruff check src/ scripts/ tests/` → 0 errors ✅
- `ruff format --check` → no format violations ✅
- `pytest -m "not slow"` → all tests pass ✅

## Definition of Done

- [x] All existing docstrings reformatted to numpy-style
- [x] `ruff format --check` passes
- [x] `ruff check` passes (0 errors)
- [x] `pytest` — all tests pass
