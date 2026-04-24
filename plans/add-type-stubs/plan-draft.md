# Plan: Add type stubs for 3rd party packages

## Problem Statement
The code in this project uses multiple third party packages.
In order to improve typing in our code base, type-stibs for the most relevant third party package(s) shall be created.
The goal is to have type stubs for those modules and symbols from third party packages which are used in the current code base.

## Baseline
- Run `uv run pyright` and record the current error/warning/information counts.
- Run `uv run mypy` and record the current error count.
- Identify which third-party imports lack type stubs (look for `reportMissingTypeStubs` in pyright output and `import-untyped` in mypy output).
- Run `uv run ruff check src/ scripts/ tests/` and record the current error count.
- Run `uv run pytest` and record how many tests pass.

## Approach

* First, determine which (untyped) third party packages have relevant (negative) impact on typing in our code base. In other words: Which third party packages should we concentrate on to create type-stubs.
* Work in phases, ordered from easiest-to-infer types to hardest-to-infer types.
* A good example of how a type-stub structure for a third party package can look like already exists in this repo's own `stubs/` directory. Study the existing stub packages there as a model for style, depth, and registration in `pyproject.toml` / `pyrightconfig.json`. Follow a similar approach when creating type-stubs for additional packages.
* For each 3rd-party class you stub, trace all attribute access chains used in the codebase (e.g. `.venv.envs[0]`). Stub every intermediate class and attribute that appears in such a chain, not only the directly imported symbols.

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
- After adding stubs, scan all source files for `# type: ignore` comments that have become unnecessary. Use pyright's `reportUnnecessaryTypeIgnoreComment` (surfaces as "information" level) and `uv run mypy --warn-unused-ignores` to detect them. Remove stale suppressions.

## Definition of Done

- [ ] Type-stubs added for the most relevant third party package (all phases complete)
- [ ] `uv run ruff --select ANN`, `uv run pyright`, and `uv run mypy` report fewer than 5 missing type annotations (target: 0)
- [ ] `uv run ruff format --check` — no format violations
- [ ] `uv run ruff check` — no regression in ruff checks
- [ ] `uv run pytest` — all tests pass
- [ ] No rules added to the `ignore` lists in `ruff.toml` and `pyproject.toml` without documented justification
