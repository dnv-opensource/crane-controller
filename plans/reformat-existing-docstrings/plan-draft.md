# Plan: Reformat existing docstrings

## Problem Statement
In the current code base, existing docstrings are formatted in various, non-uniform styles.
Goal is to reformat all existing docstrings to be numpy-style compliant.

## Proposed Approach
- Reformat existing docstrings to be numpy-style compliant.
- Ensure the `Parameters` section exists and is complete.
- Ensure that `Returns` section exists and is complete in case the documented function or method returns anything different from `None`.
- Ensure that `Raises` section exists and is complete in case the documented function or method raises exceptions.
- If you feel it is reasonable and adds clarity, you can optionally add further numpydoc-defined sections (`Warns`, `See also`, `Notes`, `References`, `Examples`).
- Eventually, improve the wording of docstrings where you feel the existing description is incorrect or outdated.

## Context
- Relevant modules: all (`src/**`, `scripts/**`)
- Related tests: None
- Constraints:
  - Do **not** use ambiguous characters in docstrings (em-dash `—`, en-dash `–`, etc.). Use ASCII equivalents (e.g. hyphen `-`). Exemptions: characters listed in `allowed-confusables` in `ruff.toml`.
  - Run `uv run ruff format` after every phase
  - Run `uv run ruff check` to monitor progress and ensure there is no regression in other ruff checks.

## Validation Strategy
- Run `uv run ruff check` to ensure there is no regression in ruff checks.
- Run `uv run ruff format --check` to ensure formatting is not broken.
- Run `uv run pytest` to ensure there is no regression in tests.

## Definition of Done
- [ ] All existing docstrings reformated to be numpy-style compliant.
- [ ] `uv run ruff format --check`
- [ ] `uv run ruff check` passes
- [ ] `uv run pytest` — all tests pass
- [ ] Documentation updated (if needed)
