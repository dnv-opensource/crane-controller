# Plan: Add missing docstrings

## Problem Statement
In the current code base, many public functions, methods, attributes, classes, modules and sub-packages are still missing docstrings.
Goal is to ensure all public functions, methods, attributes, classes, modules and sub-packages have docstrings.

## Baseline
- Run `uv run ruff check src/ scripts/` and record the current error count (should be 0 with D10x rules ignored).
- Run `uv run ruff check src/ scripts/ --select D100,D101,D102,D103,D104,D105,D107` and record the violation count per rule.
- Run `uv run ruff format --check` and confirm formatting is clean.
- Run `uv run pytest` and record how many tests pass.
- Inventory all missing docstrings by file, listing the symbol name and violation rule.

## Proposed Approach
Use `uv run ruff check` to identify missing docstrings and monitor your progress.
In ruff.toml, following ruff rules from the `D10x` family are currently deactivated, i.e. listed as "temporarily ignored":
    "D100", # Missing docstring in public module
    "D101", # Missing docstring in public class
    "D102", # Missing docstring in public method
    "D103", # Missing docstring in public function
    "D104", # Missing docstring in public package
    "D105", # Missing docstring in magic method
    "D107", # Missing docstring in __init__
Goal is to eventually have these rules reactivated again (i.e. commented out in ruff.toml).
As you progress, consider to sequentially re-activate the rules in ruff.toml, by step-by-step commenting them back out, then add the respective docstrings where flagged missing until `uv run ruff check` reports no more missing docstrings, and then repeat same for the next `D10x` rule.
Work in phases, "inside-out": Start with doctsrings for symbols defined in a smaller / more local scope to docstrings for symbols defined in a more global scope.
In each phase:
- Create docstrings where missing.
- Ensure any new docstrings are numpy-style compliant.
- Add the `Parameters` section and ensure it is complete.
- Add the `Returns` section in case the documented function or method returns anything different from `None`.
- Add the `Raises` section in case the documented function or method raises exceptions.
- If you feel it is reasonable and adds clarity, you can optionally add further numpydoc-defined sections (`Warns`, `See also`, `Notes`, `References`, `Examples`).

## Context
- Relevant modules: all (`src/**`, `scripts/**`)
- Related tests: None
- Constraints:
  - Do **not** use ambiguous characters in docstrings, like for example — (em-dash) and – (en-dash). Use unambiguous ASCII characters instead, e.g. - (hyphen). Exemptions are characters which are explicitely listed in `allowed-confusables` in ruff.toml
  - Run `uv run ruff format` after every phase
  - Run `uv run ruff check` to monitor progress and ensure there is no regression in other ruff checks.

## Validation Strategy
- Run `uv run ruff check` to ensure there is no regression in ruff checks.
- Run `uv run ruff format --check` to ensure formatting is not broken.
- Run `uv run pytest` to ensure there is no regression in tests.

## Definition of Done
- [ ] In ruff.toml, all temporarily ignored `D10x` rules are activated again, i.e. commented out again.
- [ ] All missing docstrings are added.
- [ ] `uv run ruff format --check`
- [ ] `uv run ruff check` passes
- [ ] `uv run pytest` — all tests pass
- [ ] Documentation updated (if needed)
