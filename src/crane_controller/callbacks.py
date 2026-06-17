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

        [  600,192/3,000,000]  ep_len_mean↑=100  rew/step↑=-0.046
          |  kl↓=0.0163  expl_var↑=0.346  value_loss↓=0.041  entropy~=-1.188  clip_frac↓=0.175
          |  rail_hit%↓=12%  t_min↓=0.820s  |x|↓=0.019m  |xv|↓=0.001  E↓=0.0001  |ω|↓=0.001

    - Family 1: policy performance (ep_len_mean, rew/step)
    - Family 2: PPO diagnostics (kl, expl_var, value_loss, entropy, clip_frac)
    - Family 3: task quality — rail crash rate plus mean physical end-states for survived episodes:
        - rail_hit%   truncated before max_episode_steps (crane hit the rail)
        - t_min       mean minimum-time-to-stop at episode end (s)
        - |x|         mean absolute crane position at episode end (m)
        - |xv|        mean absolute crane velocity at episode end
        - E           mean load kinetic energy at episode end
        - |ω|         mean absolute load angular velocity at episode end

    If *csv_path* is given, all rows are written to a CSV file at the end of
    training for post-training analysis and plotting. The CSV also includes
    ``policy_gradient_loss`` which is omitted from the terminal line.
    """

    def __init__(
        self,
        total_timesteps: int,
        log_interval: int = 50_000,
        csv_path: str | None = None,
        max_episode_steps: int = 1000,
    ) -> None:
        """Initialize callbacks for use in PPO Agent.

        Args:
            total_timesteps: Total training timesteps (used for the progress label).
            log_interval: Minimum timesteps between log lines (default 50 000).
            csv_path: Path to write a CSV log file at the end of training (default None).
            max_episode_steps: TimeLimit cap passed to the environment (default 1000). Used to
                distinguish rail hits (ep_len < max_episode_steps) from survived
                episodes (ep_len >= max_episode_steps).
        """
        super().__init__(verbose=0)  # pyright: ignore[reportCallIssue]
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log: int = 0
        self._csv_path = csv_path
        self._max_episode_steps = max_episode_steps
        self._rows: list[dict[str, float]] = []
        # Per-interval episode counters (reset after each log line)
        self._ep_count: int = 0
        self._rail_hits: int = 0
        self._surv_t_min_sum: float = 0.0
        self._surv_t_min_n: int = 0
        self._surv_x_pos_sum: float = 0.0
        self._surv_x_pos_n: int = 0
        self._surv_x_vel_sum: float = 0.0
        self._surv_x_vel_n: int = 0
        self._surv_energy_sum: float = 0.0
        self._surv_energy_n: int = 0
        self._surv_theta_dot_sum: float = 0.0
        self._surv_theta_dot_n: int = 0
        self._surv_theta_dev_sum: float = 0.0
        self._surv_theta_dev_n: int = 0

    def _diag(self, key: str) -> float | None:
        """Read a value from SB3's internal logger; returns None if not yet available."""
        try:
            val = self.model.logger.name_to_value.get(key)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            return float(val) if val is not None else None
        except AttributeError:
            return None

    def _on_step(self) -> bool:  # noqa: C901, PLR0912, PLR0915
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
                        t_min = info.get("t_min")  # pyright: ignore[reportUnknownMemberType]
                        x_pos = info.get("x_pos")  # pyright: ignore[reportUnknownMemberType]
                        x_vel = info.get("x_vel")  # pyright: ignore[reportUnknownMemberType]
                        energy = info.get("energy")  # pyright: ignore[reportUnknownMemberType]
                        theta_dot = info.get("theta_dot")  # pyright: ignore[reportUnknownMemberType]
                        if t_min is not None:
                            self._surv_t_min_sum += float(t_min)
                            self._surv_t_min_n += 1
                        if x_pos is not None:
                            self._surv_x_pos_sum += abs(float(x_pos))
                            self._surv_x_pos_n += 1
                        if x_vel is not None:
                            self._surv_x_vel_sum += abs(float(x_vel))
                            self._surv_x_vel_n += 1
                        if energy is not None:
                            self._surv_energy_sum += float(energy)
                            self._surv_energy_n += 1
                        if theta_dot is not None:
                            self._surv_theta_dot_sum += abs(float(theta_dot))
                            self._surv_theta_dot_n += 1
                        theta = info.get("theta")  # pyright: ignore[reportUnknownMemberType]
                        if theta is not None:
                            self._surv_theta_dev_sum += abs(float(theta) - np.pi)
                            self._surv_theta_dev_n += 1

        t: int = self.num_timesteps  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        buf = self.model.ep_info_buffer  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        if t - self._last_log >= self._log_interval and buf is not None and len(buf) > 0:
            mean_rew = float(np.mean([ep["r"] for ep in buf]))
            mean_len = float(np.mean([ep["l"] for ep in buf]))
            kl = self._diag("train/approx_kl")
            ev = self._diag("train/explained_variance")
            vl = self._diag("train/value_loss")
            ent = self._diag("train/entropy_loss")
            clip = self._diag("train/clip_fraction")
            pgl = self._diag("train/policy_gradient_loss")

            # Family 1
            line = f"[{t:>8,}/{self._total:,}]  ep_len_mean↑={mean_len:.0f}  rew/step↑={mean_rew / mean_len:+.3f}"

            # Family 2
            if any(v is not None for v in (kl, ev, vl, ent, clip)):
                parts: list[str] = []
                if kl is not None:
                    parts.append(f"kl↓={kl:.4f}")  # lower is healthier (<0.02 OK)
                if ev is not None:
                    parts.append(f"expl_var↑={ev:.3f}")  # higher is healthier (>0.5 OK)
                if vl is not None:
                    parts.append(f"value_loss↓={vl:.4f}")  # lower is healthier (<0.1 OK)
                if ent is not None:
                    parts.append(f"entropy~={ent:.3f}")  # decays toward 0 over training
                if clip is not None:
                    parts.append(f"clip_frac↓={clip:.3f}")  # lower is healthier (<0.15 OK)
                line += "  |  " + "  ".join(parts)

            # Family 3 — physical end-state means for survived episodes
            _sm = lambda s, n: s / n if n > 0 else float("nan")  # noqa: E731
            t_min_m = _sm(self._surv_t_min_sum, self._surv_t_min_n)
            x_pos_m = _sm(self._surv_x_pos_sum, self._surv_x_pos_n)
            x_vel_m = _sm(self._surv_x_vel_sum, self._surv_x_vel_n)
            energy_m = _sm(self._surv_energy_sum, self._surv_energy_n)
            theta_dot_m = _sm(self._surv_theta_dot_sum, self._surv_theta_dot_n)
            theta_dev_m = _sm(self._surv_theta_dev_sum, self._surv_theta_dev_n)

            if self._ep_count > 0:
                rail_pct = 100.0 * self._rail_hits / self._ep_count
                _f = lambda v, fmt, u="": "---" if np.isnan(v) else f"{v:{fmt}}{u}"  # noqa: E731
                line += (
                    f"  |  rail_hit%↓={rail_pct:.0f}%"
                    f"  t_min↓={_f(t_min_m, '.3f', 's')}"
                    f"  |x|↓={_f(x_pos_m, '.4f', 'm')}"
                    f"  |xv|↓={_f(x_vel_m, '.4f')}"
                    f"  E↓={_f(energy_m, '.4f')}"
                    f"  |ω|↓={_f(theta_dot_m, '.4f')}"
                    f"  |θ-π|↓={_f(theta_dev_m, '.4f')}"
                )

            tqdm.write(line)

            rail_pct_val = 100.0 * self._rail_hits / self._ep_count if self._ep_count > 0 else float("nan")

            self._rows.append(
                {
                    "t": float(t),
                    "ep_len_mean": mean_len,
                    "rew_per_step": mean_rew / mean_len,
                    "approx_kl": kl if kl is not None else float("nan"),
                    "explained_variance": ev if ev is not None else float("nan"),
                    "value_loss": vl if vl is not None else float("nan"),
                    "entropy_loss": ent if ent is not None else float("nan"),
                    "clip_fraction": clip if clip is not None else float("nan"),
                    "policy_gradient_loss": pgl if pgl is not None else float("nan"),
                    "rail_hit_pct": rail_pct_val,
                    "mean_t_min": t_min_m,
                    "mean_x_pos_abs": x_pos_m,
                    "mean_x_vel_abs": x_vel_m,
                    "mean_energy": energy_m,
                    "mean_theta_dot_abs": theta_dot_m,
                    "mean_theta_dev": theta_dev_m,
                }
            )

            # Reset per-interval counters
            self._ep_count = 0
            self._rail_hits = 0
            self._surv_t_min_sum = 0.0
            self._surv_t_min_n = 0
            self._surv_x_pos_sum = 0.0
            self._surv_x_pos_n = 0
            self._surv_x_vel_sum = 0.0
            self._surv_x_vel_n = 0
            self._surv_energy_sum = 0.0
            self._surv_energy_n = 0
            self._surv_theta_dot_sum = 0.0
            self._surv_theta_dot_n = 0
            self._surv_theta_dev_sum = 0.0
            self._surv_theta_dev_n = 0
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
