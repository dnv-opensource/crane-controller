# Plan: Fix issues raised by ruff check

## Problem Statement

`ruff check` currently raises various errors across the `src/` tree.
The goal is to reduce the error count to **below 10** (ideally zero) by systematically fixing each violation category — not by suppressing rules.

## Baseline
- Run `uv run ruff check src/ scripts/` and record the current error count and violation categories.
- Run `uv run ruff check src/ scripts/ --statistics` to get a per-rule summary.
- Run `uv run ruff format --check` and confirm formatting is clean.
- Run `uv run pytest` and record how many tests pass.

## Approach

Work in phases, ordered from easiest mechanical fixes to the most structural refactors.
After each phase: run `uv run ruff format`, `uv run ruff check`, and `uv run pytest`.

## Context

- Relevant modules: all (`src/**`, `tests/**`, `stubs/**`)
- Related tests: all (run `uv run pytest` to guard against regressions)
- Constraints:
  - Do **not** suppress rules globally in `ruff.toml`. The goal is to actually fix the code, not hide errors.
  - Suppress per-line with `# noqa` only as a last resort for cases where the lint rule is genuinely inapplicable.
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Validation Strategy

- Run `uv run ruff check` and confirm error count is below 10 (ideally zero).
- Run `uv run pytest` and confirm all tests pass.
- Run `uv run ruff format --check` to ensure formatting is not broken.

## Definition of Done

- [ ] Code implemented (all phases complete)
- [ ] `uv run ruff check` reports fewer than 10 errors (target: 0)
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff format --check` — no format violations
- [ ] No rules added to the `ignore` list in `ruff.toml` without documented justification
