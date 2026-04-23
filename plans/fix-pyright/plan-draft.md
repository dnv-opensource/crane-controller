# Plan: Fix issues raised by pyright

## Problem Statement

`pyright` currently raises various errors across the `src/` tree.
The goal is to reduce the error count to **below 5** (ideally zero) and eliminate as many warnings as practical by fixing the underlying code — not by suppressing rules.

## Approach

Work in phases, ordered from easiest mechanical fixes to the most structural refactors.
After each phase: run `uv run ruff format`, `uv run pyright`, and `uv run pytest`.

## Context

- Relevant modules: all (`src/**`, `tests/**`, `scripts/**`, `stubs/**`)
- Related tests: all (run `uv run pytest` to guard against regressions)
- Constraints:
  - Do **not** suppress rules globally in `pyproject.toml`. The goal is to actually fix the code, not hide errors.
  - Suppress per-line with `# pyright: ignore` only as a last resort for genuinely inapplicable rules.
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Validation Strategy

- Run `uv run pyright` and confirm error count is below 5 (ideally zero).
- Run `uv run pytest` and confirm all tests pass.
- Run `uv run ruff format --check` to ensure formatting is not broken.

## Definition of Done

- [ ] Code implemented (all phases complete)
- [ ] `uv run pyright` reports fewer than 5 errors (target: 0)
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff format --check` — no format violations
- [ ] No rules added to the `ignore` list in `pyproject.toml` without documented justification
