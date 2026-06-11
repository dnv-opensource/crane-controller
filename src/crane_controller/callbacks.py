"""SB3 training callbacks for the crane-controller project."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class EpRewardLogCallback(BaseCallback):
    """Logs episode stats and PPO diagnostics via tqdm.write() every log_interval steps.

    Prints one line per interval with three ``|``-separated families::

        [  600,192/3,000,000]  ep_rew_mean↑=  -27.68  ep_len_mean↑=100  rew/step↑=-0.046
          |  kl↓=0.0163  expl_var↑=0.346  value_loss↓=0.041  entropy~=-1.188  clip_frac↓=0.175
          |  rail_hit%↓=12%  timelimit%↓=70%  success%↑=18%

    - Family 1: policy performance (ep_rew_mean, ep_len_mean, rew/step)
    - Family 2: PPO diagnostics (kl, expl_var, value_loss, entropy, clip_frac)
    - Family 3: task quality — three counters that must sum to 100%:
        - rail_hit%   truncated before max_episode_steps (crane hit the rail)
        - timelimit%  survived all steps but ep_rew < success_threshold (not solved)
        - success%    survived all steps AND ep_rew >= success_threshold (solved)

    If *csv_path* is given, all rows are written to a CSV file at the end of
    training for post-training analysis and plotting. The CSV also includes
    ``policy_gradient_loss`` which is omitted from the terminal line.

    Parameters
    ----------
    total_timesteps : int
        Total training timesteps (used for the progress label).
    log_interval : int
        Minimum timesteps between log lines (default 50 000).
    csv_path : str or None
        Path to write a CSV log file at the end of training (default None).
    max_episode_steps : int
        TimeLimit cap passed to the environment (default 100). Used to
        distinguish rail hits (ep_len < max_episode_steps) from survived
        episodes (ep_len >= max_episode_steps).
    success_threshold : float
        Minimum total episode reward to count as a solved episode (default
        -50.0). Episodes that survive ``max_episode_steps`` but whose total
        reward is below this threshold are counted as ``timelimit`` (alive but
        not solved). Tune based on expected solved-episode reward:
        ``ep_rew_mean ≈ -0.07/step × 1000 steps ≈ -70`` for a good policy;
        ``-50`` is a conservative threshold that only triggers once the policy
        is clearly converging.
    """

    def __init__(
        self,
        total_timesteps: int,
        log_interval: int = 50_000,
        csv_path: str | None = None,
        max_episode_steps: int = 1000,
        success_threshold: float = -15.0,
    ) -> None:
        super().__init__(verbose=0)  # pyright: ignore[reportCallIssue]
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log: int = 0
        self._csv_path = csv_path
        self._max_episode_steps = max_episode_steps
        self._success_threshold = success_threshold
        self._rows: list[dict[str, float]] = []
        # Per-interval episode counters (reset after each log line)
        self._ep_count: int = 0
        self._rail_hits: int = 0
        self._timelimit_count: int = 0
        self._success_count: int = 0

    def _diag(self, key: str) -> float | None:
        """Read a value from SB3's internal logger; returns None if not yet available."""
        try:
            val = self.model.logger.name_to_value.get(key)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            return float(val) if val is not None else None
        except AttributeError:
            return None

    def _on_step(self) -> bool:  # noqa: C901, PLR0912, PLR0915
        # Three-bucket classification per episode:
        #   rail_hit  — truncated before max_episode_steps
        #   timelimit — survived all steps but ep_rew < success_threshold
        #   success   — survived all steps AND ep_rew >= success_threshold
        # info["r"] is the total episode reward injected by SB3's Monitor wrapper.
        _locals: dict[str, Any] = self.locals  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        dones = _locals.get("dones")  # pyright: ignore[reportUnknownMemberType]
        infos = _locals.get("infos")  # pyright: ignore[reportUnknownMemberType]
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos, strict=False):
                if done:
                    self._ep_count += 1
                    ep_steps: int = int(info.get("steps", 0))  # pyright: ignore[reportUnknownMemberType]
                    if ep_steps < self._max_episode_steps:
                        self._rail_hits += 1
                    else:
                        ep_info = info.get("episode", {})  # pyright: ignore[reportUnknownMemberType]
                        ep_rew: float = float(ep_info.get("r", float("-inf")))  # pyright: ignore[reportUnknownMemberType]
                        if ep_rew >= self._success_threshold:
                            self._success_count += 1
                        else:
                            self._timelimit_count += 1

        t: int = self.num_timesteps  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        buf = self.model.ep_info_buffer  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        if t - self._last_log >= self._log_interval and len(buf) > 0:
            mean_rew = float(np.mean([ep["r"] for ep in buf]))
            mean_len = float(np.mean([ep["l"] for ep in buf]))
            kl   = self._diag("train/approx_kl")
            ev   = self._diag("train/explained_variance")
            vl   = self._diag("train/value_loss")
            ent  = self._diag("train/entropy_loss")
            clip = self._diag("train/clip_fraction")
            pgl  = self._diag("train/policy_gradient_loss")

            # Family 1
            line = (
                f"[{t:>8,}/{self._total:,}]"
                f"  ep_rew_mean↑={mean_rew:+8.2f}"
                f"  ep_len_mean↑={mean_len:.0f}"
                f"  rew/step↑={mean_rew / mean_len:+.3f}"
            )

            # Family 2
            if any(v is not None for v in (kl, ev, vl, ent, clip)):
                parts: list[str] = []
                if kl is not None:
                    parts.append(f"kl↓={kl:.4f}")          # lower is healthier (<0.02 OK)
                if ev is not None:
                    parts.append(f"expl_var↑={ev:.3f}")     # higher is healthier (>0.5 OK)
                if vl is not None:
                    parts.append(f"value_loss↓={vl:.4f}")   # lower is healthier (<0.1 OK)
                if ent is not None:
                    parts.append(f"entropy~={ent:.3f}")     # decays toward 0 over training
                if clip is not None:
                    parts.append(f"clip_frac↓={clip:.3f}")  # lower is healthier (<0.15 OK)
                line += "  |  " + "  ".join(parts)

            # Family 3 — shown only once episodes start completing
            if self._ep_count > 0:
                n = self._ep_count
                rail_pct      = 100.0 * self._rail_hits      / n
                timelimit_pct = 100.0 * self._timelimit_count / n
                success_pct   = 100.0 * self._success_count  / n
                line += (
                    f"  |  rail_hit%↓={rail_pct:.0f}%"
                    f"  timelimit%↓={timelimit_pct:.0f}%"
                    f"  success%↑={success_pct:.0f}%"
                )

            tqdm.write(line)

            rail_pct_val      = 100.0 * self._rail_hits      / self._ep_count if self._ep_count > 0 else float("nan")
            timelimit_pct_val = 100.0 * self._timelimit_count / self._ep_count if self._ep_count > 0 else float("nan")
            success_pct_val   = 100.0 * self._success_count  / self._ep_count if self._ep_count > 0 else float("nan")

            self._rows.append({
                "t": float(t),
                "ep_rew_mean":          mean_rew,
                "ep_len_mean":          mean_len,
                "rew_per_step":         mean_rew / mean_len,
                "approx_kl":            kl   if kl   is not None else float("nan"),
                "explained_variance":   ev   if ev   is not None else float("nan"),
                "value_loss":           vl   if vl   is not None else float("nan"),
                "entropy_loss":         ent  if ent  is not None else float("nan"),
                "clip_fraction":        clip if clip is not None else float("nan"),
                "policy_gradient_loss": pgl  if pgl  is not None else float("nan"),
                "rail_hit_pct":         rail_pct_val,
                "timelimit_pct":        timelimit_pct_val,
                "success_pct":          success_pct_val,
            })

            # Reset per-interval counters
            self._ep_count = 0
            self._rail_hits = 0
            self._timelimit_count = 0
            self._success_count = 0
            self._last_log = t

        return True

    def _on_training_end(self) -> None:
        if self._csv_path and self._rows:
            Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(self._rows[0].keys())
            with Path(self._csv_path).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)
