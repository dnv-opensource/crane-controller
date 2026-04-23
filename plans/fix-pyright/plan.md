# Plan: Fix issues raised by pyright — COMPLETED

## Result

`uv run pyright src/ tests/ scripts/ stubs/` now reports **0 errors, 0 warnings, 0 informations**.

## Problem Statement

`uv run pyright src/ tests/ scripts/ stubs/` currently reports **0 errors, 77 warnings, 8 informations**.
The goal is to reduce warnings as far as practical (ideally below 5) and eliminate all informations
by fixing the underlying code — not by suppressing rules or weakening the pyright configuration.

## Context

- Relevant modules: `src/**`, `tests/**`, `scripts/**`, `stubs/**`
- Related tests: all (`uv run pytest`)
- Pyright config: `pyproject.toml` → `[tool.pyright]`, `typeCheckingMode = "basic"`, `stubPath = "stubs"`
- Constraints:
  - Do **not** suppress rules globally in `pyproject.toml`
  - Suppress per-line with `# pyright: ignore[...]` only as a last resort for genuinely inapplicable rules
  - Run `uv run ruff format` after every phase
  - Existing tests must continue to pass after every phase

## Assumptions

- The 20 custom type stubs under `stubs/` (torch, matplotlib, stable\_baselines3) are correct and stay unchanged.
- No pyright version changes during this work.
- The `scripts/` directory is included even though it is outside the pyright `include` list — the plan-draft explicitly requests it.

## Baseline (0 errors, 77 warnings, 8 informations)

### Warning breakdown by rule

| Rule | Count | Files |
|---|---|---|
| `reportUnusedCallResult` | 44 | scripts/ (28), envs/ (10), reinforce\_agent (2), crane\_factory (2), test\_environment (2) |
| `reportUnknownMemberType` | 20 | q\_agent (8), reinforce\_agent (6), analyse\_q (2), test\_q (2), test\_environment (1), q\_agent/update (1) |
| `reportUninitializedInstanceVariable` | 7 | reinforce\_agent (4), controlled\_crane\_pendulum (1), controlled\_mobile\_crane (2) |
| `reportIncompatibleMethodOverride` | 2 | clip\_reward (1), relative\_position (1) |
| `reportUnnecessaryComparison` | 1 | controlled\_mobile\_crane (1) |

### Information breakdown

| Rule | Count | Files |
|---|---|---|
| `reportUnnecessaryTypeIgnoreComment` | 7 | algorithm (4), train\_ppo (2), play\_ppo (1) |
| `reportUnnecessaryIsInstance` | 1 | q\_agent (1) |

## Proposed Approach

Work in phases ordered from mechanical/safe fixes to structural ones.
After each phase: run `uv run ruff format`, `uv run pyright src/ tests/ scripts/ stubs/`, and `uv run pytest`.

---

### Phase 1 — Remove unnecessary `# type: ignore` comments (8 informations → 0)

**Goal:** Eliminate all 8 informations.

**Files to edit:**

| File | Lines | Comment to remove |
|---|---|---|
| `algorithm.py` | 64, 133, 134, 167 | 4× `# type: ignore[attr-defined]` — attributes now resolve correctly |
| `scripts/train_ppo.py` | 42, 55 | 2× `# type: ignore[arg-type]` — `AntiPendulumEnv` is now compatible |
| `scripts/play_ppo.py` | 26 | 1× `# type: ignore[arg-type]` — same as above |
| `q_agent.py` | 195 | Remove unnecessary `isinstance` check (or the guard it protects) |

**Scope:** Comment-only changes (plus one possible guard removal). Zero behavioral impact.

**Validation:** informations drop from 8 → 0.

---

### Phase 2 — Assign discarded return values to `_` (44 warnings → 0)

**Goal:** Eliminate all 44 `reportUnusedCallResult` warnings.

These are calls whose return values are intentionally ignored (e.g. `env.reset()`, `plt.title()`, `pygame.draw.*`).
The idiomatic fix is `_ = call(...)`.

**Files to edit (by location):**

| File | Count | Typical calls |
|---|---|---|
| `scripts/analyse_q.py` | 2 | `Action` returns |
| `scripts/play_ppo.py` | 3 | `Action` returns |
| `scripts/play_q.py` | 5 | `Action` + `env.step` |
| `scripts/train_ppo.py` | 5 | `Action` returns |
| `scripts/train_q.py` | 8 | `Action` + `env.step` |
| `controlled_crane_pendulum.py` | 7 | `env.reset`, `plt.title`, `plt.xlim/ylim`, `np.clip` |
| `controlled_mobile_crane.py` | 10 | `env.reset`, `pygame.draw.rect/circle/line` |
| `crane_factory.py` | 2 | `Boom(...)` side-effect constructors |
| `reinforce_agent.py` | 2 | `loss.backward()`, `torch.manual_seed()` |
| `test_environment.py` | 2 | `plt.title`, `np.clip` return |

**Pattern:** Prefix each call with `_ = ` (or in some cases, unpack into `_, info = env.reset()`).

**Scope:** Purely cosmetic — adds `_ = ` prefix. Zero behavioral impact.

**Validation:** `reportUnusedCallResult` warnings drop from 44 → 0.

---

### Phase 3 — Declare uninitialized instance variables (7 warnings → 0)

**Goal:** Eliminate all 7 `reportUninitializedInstanceVariable` warnings.

Instance variables that are first assigned in `reset()` or a non-`__init__` method must be declared
in the class body or `__init__` with a type annotation (possibly set to a sentinel or `None`).

**Files to edit:**

| File | Variable | Fix |
|---|---|---|
| `reinforce_agent.py` | `probs` (L109) | Add `self.probs: list[Tensor] = []` in `__init__` (then `reset` reinitializes) |
| `reinforce_agent.py` | `rewards` (L110) | Add `self.rewards: list[float] = []` in `__init__` |
| `reinforce_agent.py` | `net` (L112) | Add `self.net: PolicyNetwork` declaration in `__init__` — call `self.reset()` from `__init__` |
| `reinforce_agent.py` | `optimizer` (L113) | Same — initialized by `reset()`, needs declaration or `reset()` call |
| `controlled_crane_pendulum.py` | `steps` (L331) | Add `self.steps: int = 0` in `__init__` |
| `controlled_mobile_crane.py` | `_agent_location` (L112) | Add typed declaration in `__init__` |
| `controlled_mobile_crane.py` | `_target_location` (L116) | Add typed declaration in `__init__` |

**Risk:** Calling `self.reset()` from `__init__` to initialize `net`/`optimizer` changes initialization order.
Verify `reset()` has no dependencies on other `__init__` state that would break.

**Validation:** `reportUninitializedInstanceVariable` warnings drop from 7 → 0.

---

### Phase 4 — Type-narrow `q_values` to eliminate `Unknown` union (20 warnings → target ≤ 2)

**Goal:** Eliminate most of the 20 `reportUnknownMemberType` warnings caused by `q_values` having a
`defaultdict[..., ...] | defaultdict[Unknown, ...]` union type.

**Root cause:** `q_values` is assigned in two branches — one from `defaultdict(lambda: ...)` (fully typed)
and one from `read_dumped()` (returns `defaultdict[tuple[int, ...], np.ndarray]`). The union of these
produces `Unknown` in one branch because `read_dumped` uses a lambda with an untyped `self.env.action_space.n`.

**Fix options (choose one):**

1. **Explicit annotation on `self.q_values`:**
   ```python
   self.q_values: defaultdict[tuple[int, ...], np.ndarray] = ...
   ```
   This narrows the union away. The lambda's inferred default factory still works because `np.ndarray`
   is the value type in both branches.

2. **Type the lambda explicitly** in both `__init__` and `read_dumped`:
   ```python
   factory: Callable[[], np.ndarray] = lambda: np.array(...)
   self.q_values = defaultdict(factory)
   ```

Option 1 is simpler. Apply to both `q_agent.py` (2 sites) and verify the cascade through
`analyse_q.py`, `test_q.py`, and `test_environment.py`.

**Also fix:** The `update` method warning (L207) — `dict.update` inherits `Unknown` key type from
the `q_values` union. Fixing the root annotation should resolve this transitively.

**Remaining `list[Unknown]` warnings in `reinforce_agent.py`:** These 6 warnings stem from
`self.probs = []` and `self.rewards = []` being untyped empty lists. Phase 3 adds type annotations
(`list[Tensor]`, `list[float]`), which should resolve these as a side effect.

**Validation:** `reportUnknownMemberType` warnings drop from 20 → ≤ 2.

---

### Phase 5 — Fix incompatible method overrides (2 warnings)

**Goal:** Eliminate the 2 `reportIncompatibleMethodOverride` warnings.

**`clip_reward.py` — `ClipReward.reward`:**
- Base class `RewardWrapper.reward()` expects `reward: SupportsFloat`.
- Override uses `reward: float`.
- **Fix:** Widen parameter type to `SupportsFloat` and cast to `float` inside the body.

**`relative_position.py` — `RelativePosition.observation`:**
- Base class `ObservationWrapper.observation()` expects `observation: np.ndarray`, returns `np.ndarray`.
- Override expects `dict[str, np.ndarray]`, returns `np.ndarray`.
- The parameter type is wrong: the actual observation arriving from the base env is a `dict`, but
  the declared base type is `np.ndarray`. The wrapper's generic parameter already says
  `ObservationWrapper[dict[str, np.ndarray], ...]` so the override should match.
- **Fix:** Change the parameter type to `np.ndarray` (matching the base) if the env really produces
  arrays, or suppress with a targeted `# pyright: ignore[reportIncompatibleMethodOverride]` with
  a comment explaining the gymnasium API mismatch. Investigate at runtime first.

**Validation:** `reportIncompatibleMethodOverride` warnings drop from 2 → 0.

---

### Phase 6 — Fix unnecessary comparison (1 warning)

**Goal:** Eliminate the `reportUnnecessaryComparison` in `controlled_mobile_crane.py:66`.

**Root cause:** `self.mode` is typed as `Literal[1, 2]` (from the constructor), making `self.mode == 0`
always `False`.

**Fix options:**
1. Widen `mode` type to `int` if 0 is a valid runtime value.
2. Remove the dead `if self.mode == 0:` branch if mode 0 is genuinely unreachable.
3. If mode 0 is a valid configuration, fix the constructor's type annotation.

Investigate the constructor and call sites to choose the right option.

**Validation:** `reportUnnecessaryComparison` warning drops from 1 → 0.

---

### Phase 7 — Final validation & cleanup

1. Run full validation suite:
   ```
   uv run ruff format --check
   uv run ruff check src/ tests/ scripts/
   uv run pyright src/ tests/ scripts/ stubs/
   uv run pytest --tb=short -q
   ```
2. Update `CHANGELOG.md` — add entry under **Unreleased**.

## Alternatives Considered

| Alternative | Rationale for Rejection |
|---|---|
| **Disable `reportUnusedCallResult` globally** | Hides real bugs where return values carry error info; violates project rules against global suppression. |
| **Add `# pyright: ignore` on every line** | Scales poorly, masks genuine issues, violates project instructions. |
| **Skip `q_values` narrowing** | Leaves 20 warnings — nearly a third of the total — and prevents reaching the <5 target. |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **`_ = call(...)` changes could mask future bugs** | Low — these are deliberate fire-and-forget calls | Review each site; only apply where the return value is genuinely unused |
| **Calling `reset()` from `__init__` in reinforce\_agent** | Could change behavior if `reset()` depends on not-yet-set attributes | Read `reset()` carefully; if needed, inline the assignments instead of calling `reset()` |
| **`q_values` type annotation may be too narrow** | Could cause new errors if some code path stores non-tuple keys | Validate by running full pyright + pytest after the change |
| **Wrapper method override fixes may change runtime behavior** | Could introduce subtle type coercion | Test the wrappers (they are exercised by existing tests) and keep changes minimal |

## Commit Strategy

One commit per phase for clean bisectability:

1. `chore: remove stale type-ignore comments and unnecessary isinstance`
2. `chore: assign unused call results to _`
3. `fix: declare uninitialized instance variables`
4. `fix: type-narrow q_values defaultdict to eliminate Unknown`
5. `fix: correct wrapper method override signatures`
6. `fix: remove dead mode==0 branch (or widen mode type)`
7. `docs: update changelog for pyright fixes`

## Validation Strategy

- After each phase: `uv run ruff format`, `uv run pyright src/ tests/ scripts/ stubs/`, `uv run pytest`
- After Phase 7: full suite including `ruff check`

## Definition of Done

- [ ] `uv run pyright src/ tests/ scripts/ stubs/` reports 0 errors, <5 warnings, 0 informations
- [ ] `uv run pytest` — all tests pass
- [ ] `uv run ruff format --check` — no format violations
- [ ] `uv run ruff check src/ tests/ scripts/` — no lint violations
- [ ] No rules added to the `ignore` list in `pyproject.toml` without documented justification
- [ ] `CHANGELOG.md` updated
