# Plan: Fix issues raised by mypy

## Problem Statement

`mypy` currently raises various errors across the `src/` tree.
The goal is to reduce the error count to **below 5** (ideally zero) and eliminate as many warnings as practical by fixing the underlying code — not by suppressing rules.

## Baseline
- Run `uv run mypy` and record the current error count.
- Run `uv run mypy --warn-unused-ignores` and record stale `# type: ignore` comments. These often represent the bulk of the work.
- Categorize errors by error code (e.g. `arg-type`, `assignment`, `override`, `attr-defined`).
- Run `uv run ruff check src/ scripts/ tests/` and record the current error count.
- Run `uv run ruff format --check` and confirm formatting is clean.
- Run `uv run pytest` and record how many tests pass.

## Approach

Work in phases, ordered from easiest mechanical fixes to the most structural refactors.
After each phase: run `uv run ruff format`, `uv run mypy`, and `uv run pytest`.

### pyright / mypy divergence on `# type: ignore`

`# type: ignore` suppresses diagnostics for **both** mypy and pyright. `# pyright: ignore` suppresses only pyright. When mypy no longer needs a `# type: ignore` but pyright still does, replace it with `# pyright: ignore[rule]`. Always run both `uv run mypy --warn-unused-ignores` and `uv run pyright` after each change to avoid breaking one tool while fixing the other.

## Context

- Relevant modules: all (`src/**`, `tests/**`, `scripts/**`, `stubs/**`)
- Related tests: all (run `uv run pytest` to guard against regressions)
- Constraints:
  - Do **not** suppress rules globally in `pyproject.toml`. The goal is to actually fix the code, not hide errors.
  - Suppress per-line with `# type: ignore` only as a last resort for genuinely inapplicable rules.
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Validation Strategy

- Run `uv run mypy` and confirm error count is below 5 (ideally zero).
- Run `uv run mypy --warn-unused-ignores` to detect `# type: ignore` comments that have become unnecessary (e.g. after fixing stubs or annotations). Remove stale suppressions.
- Run `uv run pytest` and confirm all tests pass.
- Run `uv run ruff format --check` to ensure formatting is not broken.

## Definition of Done

- [ ] Code implemented (all phases complete)
- [ ] `uv run mypy` reports fewer than 5 errors (target: 0)
- [ ] `uv run mypy --warn-unused-ignores` reports no unused ignore comments
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff format --check` — no format violations
- [ ] No rules added to the `ignore` list in `pyproject.toml` without documented justification
