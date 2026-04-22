# GitHub Copilot – Project Instructions

This repository uses a **hybrid instruction model**:

- **Portable, tool-agnostic rules live in the repository root**
- **GitHub-specific scoping and activation live under `.github/`**

GitHub Copilot and GitHub Agents must treat the files below as the
authoritative source of project behavior.

---

## Primary Instruction Sources (Authoritative)

Always load and follow these files first:

- `/.instructions.md`
  Project-wide engineering rules, quality standards, and constraints.

- `/.prompt.md`
  Default working style and agent execution discipline.

- `/plans/plan.md` or `/plans/plan-draft.md` (when present)
  Explicit plan / orchestration artifact for Plan Mode workflows.
  The presence of `/plans/plan.md` or `/plans/plan-draft.md` indicates that a plan-first workflow is expected.
  `/plans/.plan.md` serves as a template and should not be modified directly.

These files form the **knowledge contract** for this repository and
are reviewed and maintained like production code.

---

## GitHub-Scoped Instructions (Additive)

Additional GitHub-specific guidance MAY apply from:

- `.github/instructions/*.instructions.md`
  Narrow, scoped rules using `applyTo` (must not contradict root rules).

- `.github/prompts/*.prompt.md`
  Reusable GitHub workflows (feature implementation, reviews, refactors).

- `.github/agents/*.agent.md`
  Named agent personas, if explicitly activated.

GitHub-scoped files **refine or specialize** behavior.
They must never redefine or override the intent of `/.instructions.md`.

---

## Precedence & Conflict Resolution

If multiple instructions apply, resolve conflicts in this order:

1. Scoped `.github/instructions/*.instructions.md` (most specific — may only **narrow or refine** root rules, never override or contradict them)
2. `/.instructions.md`
3. `/.prompt.md`

When in doubt:
- Prefer correctness, testability, and clarity over speed
- Ask for clarification rather than guessing

---

## Safety & Boundaries

- Work only within this repository
- Do not assume network or internet access unless explicitly stated
- Do not introduce hidden side effects, hooks, or background behavior
- Do not change unrelated code opportunistically

All agent output is subject to human review and normal PR processes.

---

## Agent Operating Mode

Use a **plan-first workflow** by default:

1. Restate the task and assumptions
2. Produce or update `plan.md` (unless instructed otherwise)
3. Wait for confirmation if scope is ambiguous
4. Implement incrementally
5. Verify with tests
6. Summarize changes and remaining risks

This discipline is mandatory for non-trivial changes.

---

## Intentional Non-Goals

- This file does **not** contain detailed coding rules
  (those belong in `/.instructions.md`)
- This file does **not** encode organization policy or governance logic
- This file avoids duplication by design

Less is intentional. Stability is a feature.
