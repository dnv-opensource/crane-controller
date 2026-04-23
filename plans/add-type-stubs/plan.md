# Plan: Add type stubs for third-party packages

## 0. Instruction Sources and Precedence

This plan follows `.github/copilot-instructions.md`:

- Primary instruction sources loaded: `/.instructions.md`, `/.prompt.md`, and `plans/add-type-stubs/plan-draft.md`.
- Scoped `.github/instructions/*.instructions.md` were checked; none are specific to this task.
- Conflict handling uses the repository precedence model (scoped → `/.instructions.md` → `/.prompt.md`).
- Plan-first workflow is applied for this non-trivial maintenance task.

## 1. Problem Statement (restated)

Third-party packages used in this project ship incomplete type stubs,
causing pyright to report `reportUnknownMemberType` warnings wherever
their APIs surface `Unknown` parameters or return types.
These `Unknown` types propagate through call chains (cascading), degrading
the type-safety of the project's own code.

The goal is to create **local type stubs** in `stubs/` for the most
impactful third-party packages, following the same pattern used by the
`C:/Dev/sim-orchestrator/stubs/fmpy-stubs/` reference implementation.

## 2. Current Baseline

| Metric | Value |
|---|---|
| `ruff check --select ANN` | 0 violations |
| `ruff check` | 0 violations |
| `ruff format --check` | clean |
| `pyright` errors | **0** |
| `pyright` warnings | **106** |
| `pyright` informations | **15** |
| `mypy` errors | **12** (in 7 files) |
| `pytest` | **7 passed** |

### Stub Infrastructure

Already configured:

- `stubs/` directory exists (currently empty).
- `pyproject.toml` → `[tool.pyright] stubPath = "stubs"` (stubs override inline package types).
- `pyproject.toml` → `[tool.mypy] mypy_path = "stubs"`.

### Warning Breakdown — `reportUnknownMemberType` (46 warnings)

| Source Package | Symbols with `Unknown` | Direct | Cascading | Total |
|---|---|---|---|---|
| **matplotlib.pyplot** | `subplots`, `show`, `plot`, `title`, `ion`, `xlim`, `ylim` | 21 | 0 | **21** |
| **torch** | `from_numpy`, `backward`, `step`, `manual_seed` | 4 | 8 | **12** |
| internal (`q_values` union) | `q_values`, `update` | 12 | 0 | **12** |
| **stable_baselines3** | `load`, `learn` | 2 | 0 | **2** |

Notes:

- The 12 `q_values` warnings are **not fixable via stubs** — they stem from a
  `defaultdict[tuple[int, ...], ...] | defaultdict[Unknown, ...]` union produced
  by JSON deserialization in `QLearningAgent`. These require code-level narrowing,
  not stubs.
- The remaining 60 warnings (reportUnusedCallResult, reportUninitializedInstanceVariable,
  reportIncompatibleMethodOverride, etc.) are unrelated to stubs.

### Package Stub Status

All three target packages already ship `py.typed` markers with inline stubs,
but those stubs contain `Unknown` parameters:

| Package | `py.typed` | Gap |
|---|---|---|
| `torch` 2.10.0 | Yes | `from_numpy(ndarray: Unknown)`, `backward(gradient: Unknown)`, `step(closure: Unknown)`, `manual_seed(seed: Unknown)` |
| `matplotlib` 3.10+ | Yes | `subplots(**fig_kw: Unknown)`, `show(...)`, `plot(data: Unknown)`, `title(**kwargs: Unknown)`, `ion()`, `xlim(...)`, `ylim(...)` |
| `stable_baselines3` 2.8+ | Yes | `PPO.load(**kwargs: Unknown)`, `PPO.learn(callback: (...) -> Unknown)` |

## 3. Assumptions

- Stubs in `stubs/<package>-stubs/` take the **highest priority** in pyright's
  resolution order, overriding the package's inline stubs for covered modules.
- Uncovered submodules should **fall back** to the package's inline stubs.
  This assumption will be validated in Phase 0.
- Only symbols actually imported in `src/`, `tests/`, and `scripts/` need stubs.
- The reference structure in `C:/Dev/sim-orchestrator/stubs/fmpy-stubs/` is
  the canonical model (PEP 561 naming, `.pyi` files, `...` bodies).

## 4. Proposed Approach

### Phase 0 — Validate Stub Override Mechanics

**Goal:** Confirm that partial stubs in `stubPath` override only the modules
they cover, with remaining modules falling back to inline package stubs.

**Steps:**

1. Create a minimal `stubs/torch-stubs/__init__.pyi` containing just `from_numpy`.
2. Run `uv run pyright src/crane_controller/reinforce_agent.py` and check:
   - The `from_numpy` warning is gone.
   - `torch.nn`, `torch.optim`, `torch.distributions` still resolve normally.
3. If submodules break, pivot to providing complete re-exports for every
   submodule the codebase uses.
4. Remove the test stub.

**Deliverable:** Confirmed resolution behaviour; documented in this plan.

### Phase 1 — torch Stubs (highest impact per symbol)

**Goal:** Eliminate 12 `reportUnknownMemberType` warnings (4 direct + 8 cascading).

**Rationale:** torch has the best ROI — 4 symbol stubs eliminate 12 warnings.
The `from_numpy(ndarray: Unknown)` gap cascades into `list[Unknown]` for
`probs` and `rewards`, polluting `append()` calls downstream.

**Files to create:**

```
stubs/
  torch-stubs/
    __init__.pyi        # from_numpy, manual_seed, Tensor, log, exp, tensor
    nn/
      __init__.pyi      # Module, Sequential, Linear, Tanh
    optim/
      __init__.pyi      # AdamW (Optimizer base)
    distributions/
      normal.pyi        # Normal
```

**Symbols to stub (derived from actual imports in `src/crane_controller/reinforce_agent.py`):**

| Module | Symbols | Signature Gaps to Fix |
|---|---|---|
| `torch` | `from_numpy`, `manual_seed`, `Tensor`, `log`, `exp`, `tensor` | `from_numpy(ndarray: ndarray) -> Tensor`, `manual_seed(seed: int) -> Generator` |
| `torch.nn` | `Module`, `Sequential`, `Linear`, `Tanh` | Class definitions with `forward()`, `__call__()` |
| `torch.optim` | `AdamW` | `step(closure: Callable[..., Tensor] \| None = None) -> Tensor \| None`, `zero_grad()` |
| `torch.distributions.normal` | `Normal` | `sample() -> Tensor`, `log_prob(value: Tensor) -> Tensor` |

**Also used in `tests/conftest.py`:**

| Module | Symbols |
|---|---|
| `torch` | `cuda.is_available`, `set_default_device` |
| `torch.cuda` | `is_available` |

**Validation:**

```
uv run pyright src/crane_controller/reinforce_agent.py   # from_numpy, backward, step, manual_seed warnings gone
uv run pyright tests/conftest.py                          # torch.cuda still resolves
uv run pyright src/ tests/ scripts/                       # full run — 12 fewer warnings
uv run pytest --tb=short -q                               # no regressions
```

### Phase 2 — matplotlib.pyplot Stubs (highest absolute count)

**Goal:** Eliminate 21 `reportUnknownMemberType` warnings.

**Files to create:**

```
stubs/
  matplotlib-stubs/
    __init__.pyi            # (minimal — may only re-export pyplot)
    pyplot.pyi              # subplots, show, plot, title, ion, xlim, ylim, legend, rcParams
```

**Symbols to stub (derived from actual imports across 6 files):**

| Symbol | Current Gap | Proposed Signature |
|---|---|---|
| `subplots` | `**fig_kw: Unknown` | Overloaded; replace `Unknown` kwargs with `Any` |
| `show` | `(...) -> None` | `(block: bool \| None = None) -> None` |
| `plot` | `data: Unknown \| None` | Replace `Unknown` with `Any` |
| `title` | `**kwargs: Unknown` | `(label: str, fontdict: dict[str, Any] \| None = ..., loc: ... = ..., pad: ... = ..., *, y: ... = ..., **kwargs: Any) -> Text` |
| `ion` | Unknown return | `() -> AbstractContextManager[bool \| None, bool \| None]` |
| `xlim` / `ylim` | `(...) -> tuple[float, float]` | Properly typed overloads |
| `legend` | Not currently warned | Include for completeness |
| `rcParams` | Not currently warned | `RcParams` (dict-like) |

**Validation:**

```
uv run pyright src/ tests/ scripts/                       # 21 fewer warnings
uv run pytest --tb=short -q                               # no regressions
```

### Phase 3 — stable_baselines3 Stubs (small scope)

**Goal:** Eliminate 2 `reportUnknownMemberType` warnings.

**Files to create:**

```
stubs/
  stable_baselines3-stubs/
    __init__.pyi                    # PPO re-export
    ppo.pyi                         # PPO class with load(), learn(), predict()
    common/
      __init__.pyi
      env_util.pyi                  # make_vec_env
      evaluation.pyi                # evaluate_policy
```

**Symbols to stub (derived from `src/crane_controller/ppo_agent.py`):**

| Module | Symbols |
|---|---|
| `stable_baselines3` | `PPO` |
| `stable_baselines3.common.env_util` | `make_vec_env` |
| `stable_baselines3.common.evaluation` | `evaluate_policy` |

**Validation:**

```
uv run pyright src/crane_controller/ppo_agent.py          # load, learn warnings gone
uv run pyright src/ tests/ scripts/                       # full run
uv run pytest --tb=short -q                               # no regressions
```

### Phase 4 — Final Validation & Cleanup

**Steps:**

1. Run full validation suite:
   ```
   uv run ruff format --check
   uv run ruff check src/ tests/ scripts/
   uv run pyright src/ tests/ scripts/
   uv run mypy src/ tests/
   uv run pytest --tb=short -q
   ```
2. Remove any stale `# type: ignore` comments that are now unnecessary
   (pyright reports these as `reportUnnecessaryTypeIgnoreComment`).
3. Update `CHANGELOG.md` — add entry under **Unreleased**.
4. Commit with descriptive messages.

## 5. Alternatives Considered

| Alternative | Rationale for Rejection |
|---|---|
| **Install community stub packages** (e.g., `torch-stubs`, `matplotlib-stubs`) | No actively maintained, pyright-compatible stub packages exist for these libraries at the required versions. |
| **Suppress warnings globally** (`reportUnknownMemberType = false`) | Hides real type issues in our own code; violates project quality standards. |
| **Use `# type: ignore` on every call site** | Scales poorly, hides genuine errors, and violates project instructions against ungoverned suppression. |
| **Only stub torch** (skip matplotlib and SB3) | Would leave 23 warnings unaddressed. Phase 2 and 3 are incremental and low-risk. |

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Stubs shadow entire package** — providing `torch-stubs/__init__.pyi` may make pyright ignore torch's inline stubs for submodules we didn't cover | Unresolved imports for `torch.nn`, `torch.cuda`, etc. | Phase 0 explicitly validates this. If confirmed, provide minimal re-export stubs for all used submodules. |
| **Stub signatures drift from package updates** | Future torch/matplotlib/SB3 upgrades could introduce new parameters our stubs don't cover | Add a comment in each `.pyi` noting the package version stubbed. Review stubs when bumping dependency versions. |
| **Overriding `show()` / `subplots()` signatures could introduce false negatives** | Pyright might miss genuine type errors at call sites | Keep stub signatures conservative — prefer `Any` over omitting params, and verify against upstream signatures. |
| **mypy resolution differs from pyright** | mypy uses `mypy_path` differently; stubs might not work for both tools | Validate with both `pyright` and `mypy` after each phase. |

## 7. Commit Strategy

| Commit | Scope |
|---|---|
| `type: add torch type stubs` | Phase 0 + Phase 1 (torch-stubs) |
| `type: add matplotlib type stubs` | Phase 2 (matplotlib-stubs) |
| `type: add stable-baselines3 type stubs` | Phase 3 (stable_baselines3-stubs) |
| `chore: remove stale type-ignore comments` | Phase 4 cleanup |
| `docs: update changelog for type stubs` | Phase 4 changelog |

## 8. Definition of Done

- [ ] Type stubs created for torch, matplotlib.pyplot, and stable_baselines3
- [ ] `uv run pyright src/ tests/ scripts/` — `reportUnknownMemberType` warnings reduced by ≥ 35 (from 46 to ≤ 11)
- [ ] `uv run ruff check --select ANN src/ tests/ scripts/` — 0 violations
- [ ] `uv run ruff format --check` — no format violations
- [ ] `uv run ruff check` — no regression in ruff checks
- [ ] `uv run pytest` — all tests pass
- [ ] No rules added to `ignore` lists in `ruff.toml` or `pyproject.toml` without documented justification
- [ ] CHANGELOG.md updated
