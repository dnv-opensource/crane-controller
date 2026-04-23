# Plan: Add type stubs for 3rd party packages

## Problem Statement
The code in this project uses multiple third party packages.
In order to improve typing in our code base, type-stibs for the most relevant third party package(s) shall be created.
The goal is to have type stubs for those modules and symbols from third party packages which are used in the current code base.

## Approach

* First, determine which (untyped) third party packages have relevant (negative) impact on typing in our code base. In other words: Which third party packages should we concentrate on to create type-stubs.
* Work in phases, ordered from easiest-to-infer types to hardest-to-infer types.
* A good example of how a type-stub structure for a third party package can look like is contained in the following project: `C:/Dev/sim-orchestrator`. Therein, in sub-folder `/stubs`, you will find an example of type-stubs (in that case created for the third party package `FMPy`). Follow a similar approach when creating type-stubs in the current code base.

## Context

- Relevant modules: all (`src/**`, `tests/**`, `scripts/**`)
- Related tests: all (run `uv run pytest` to guard against regressions)
- Constraints:
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Validation Strategy

- Run `uv run ruff format --check` to ensure formatting is not broken.
- Run `uv run ruff check` to ensure no regressions in ruff checks.
- Run `uv run ruff --select ANN`, `uv run pyright`, and `uv run mypy` to monitor, and validate that number of missing type hints in the code is decreasing.

## Definition of Done

- [ ] Type-stubs added for the most relevant third party package (all phases complete)
- [ ] `uv run ruff --select ANN`, `uv run pyright`, and `uv run mypy` report fewer than 5 missing type annotations (target: 0)
- [ ] `uv run ruff format --check` — no format violations
- [ ] `uv run ruff check` — no regression in ruff checks
- [ ] `uv run pytest` — all tests pass
- [ ] No rules added to the `ignore` lists in `ruff.toml` and `pyproject.toml` without documented justification
