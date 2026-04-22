# Plan: Fix `ruff check` violations below threshold

## 0. Instruction Sources and Precedence

This plan explicitly follows `.github/copilot-instructions.md`:

- Primary instruction sources loaded: `/.instructions.md`, `/.prompt.md`, and `plans/fix-ruff-check/plan-draft.md`.
- Scoped `.github/instructions/*.instructions.md` were checked; none are present for this task.
- Conflict handling uses the repository precedence model (scoped instructions, then `/.instructions.md`, then `/.prompt.md`).
- Plan-first workflow is applied for this non-trivial maintenance task.

## 1. Restated Goal

Bring the codebase to fewer than 10 `ruff check` violations (target: 0) by fixing root causes across the repository, while preserving behavior and keeping tests green.

## 2. Scope and Constraints

- In scope: Python source and tests (`src/**`, `tests/**`, plus any additional lint-targeted paths configured in Ruff).
- Out of scope: Unrelated feature work, large architectural refactors not required to satisfy lint rules.
- Do not add broad rule suppressions in `ruff.toml`.
- Use `# noqa` only as a last resort, and only with a specific rule code and short justification comment.
- Maintain deterministic test behavior and avoid hidden side effects.

## 3. Working Assumptions

- Toolchain commands are available through `uv`.
- Current branch is dedicated to setup/maintenance updates, so lint-focused changes are expected.
- Existing tests represent the current behavioral contract and should continue to pass.

## 4. Execution Plan

### Phase A: Baseline and Categorization

1. Run `uv run ruff check` and capture full diagnostics.
2. Group findings by category and file hot spots:
   - import ordering/unused imports
   - formatting-adjacent lint
   - complexity/style issues
   - correctness/safety-related lint
3. Record initial metrics:
   - total violations
   - top rule codes by frequency
   - top files by violation count

Exit criteria:
- A prioritized fix list exists, ordered by impact and safety.

### Phase B: Mechanical, Low-Risk Fixes First

1. Apply safe automated fixes where available (`ruff check --fix` when compatible with project policy).
2. Remove unused imports/variables and resolve straightforward style issues.
3. Run `uv run ruff format`.

Validation gate:
- `uv run ruff check`
- `uv run pytest`

Exit criteria:
- Mechanical categories significantly reduced with no test regressions.

### Phase C: Targeted Manual Refactors

1. Address remaining violations requiring logic-aware edits (e.g., control flow simplification, exception handling patterns, naming consistency where lint-required).
2. Keep each change focused and minimal per file/function.
3. Add or adjust tests only when a lint-driven refactor could alter behavior.

Validation gate after each focused batch:
- `uv run ruff format`
- `uv run ruff check`
- `uv run pytest`

Exit criteria:
- Remaining lint findings are near threshold and well understood.

### Phase D: Final Pass and Exception Review

1. Resolve final violations to reach <10 (target 0).
2. If any violation is intentionally retained, document rationale inline (specific `noqa` with rule code) and ensure it is truly unavoidable.
3. Re-run full quality checks.

Final validation gate:
- `uv run ruff format --check`
- `uv run ruff check`
- `uv run pytest`

Exit criteria:
- Lint count meets target.
- Tests pass.
- Formatting check passes.

## 5. Change Management and Safety

- Make incremental commits by category to simplify review and rollback.
- Avoid touching unrelated files.
- Prefer explicit, readable fixes over clever compact rewrites.

## 6. Deliverables

- Code updates fixing Ruff violations.
- Passing verification command outputs for lint, format-check, and tests.
- `CHANGELOG.md` updated in the `Unreleased` section with a concise summary of lint-maintenance changes.
- Brief change summary with:
  - key rule categories fixed
  - any intentional exceptions and justification
  - residual risks (if any)

## 7. Definition of Done

- [ ] `uv run ruff check` reports fewer than 10 violations (target: 0)
- [ ] `uv run pytest` passes
- [ ] `uv run ruff format --check` passes
- [ ] No global lint-rule suppression added without explicit justification
- [ ] Any `noqa` use is minimal, specific, and justified inline
- [ ] `CHANGELOG.md` has an `Unreleased` entry for the maintenance changes
