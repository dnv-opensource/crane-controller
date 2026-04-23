# Plan: Add type annotations

## Problem Statement
The code base in this project is not fully typed.
The goal is to add type annotations where missing to:
* public classes
* arguments and return types of public functions, methods, and properties
* public attributes and variables
* private classes
* arguments and return types of private functions, methods, and properties
* private attributes and variables

## Baseline
- Run `uv run ruff check src/ scripts/ tests/ --select ANN` and record the current violation count per rule.
- Run `uv run ruff check src/ scripts/ tests/` and record the current error count.
- Run `uv run ruff format --check` and confirm formatting is clean.
- Run `uv run pytest` and record how many tests pass.

## Approach

Work in phases, ordered from easiest-to-resolve types to hardest-to-resolve types.

## Context

- Relevant modules: all (`src/**`, `tests/**`, `scripts/**`)
- Related tests: all (run `uv run pytest` to guard against regressions)
- Constraints:
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Validation Strategy

- Run `uv run ruff --select ANN`, first to yield the baseline, and then after each phase as a monitoring metric to validate that number of missing type annotations is decreasing.
- Run `uv run ruff format --check` to ensure formatting is not broken.
- Run `uv run ruff check` to ensure no regressions in ruff checks.

## Definition of Done

- [ ] Type annotations added (all phases complete)
- [ ] `uv run ruff --select ANN` reports fewer than 5 missing type annotations (target: 0)
- [ ] `uv run ruff format --check` — no format violations
- [ ] `uv run ruff check` — no regression in ruff checks
- [ ] `uv run pytest` — all tests pass
- [ ] No rules added to the `ignore` lists in `ruff.toml` and `pyproject.toml` without documented justification
