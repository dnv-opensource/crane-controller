"""Plot speed-sweep results from one or more *_play_results.csv files.

Examples
--------
.. code-block:: bash

    # 3-panel comparison (two CSVs):
    uv run python scripts/plot_sweep.py models/hybrid_cv01_s5775_play_results.csv models/sig_t_min_s5775_play_results.csv

    # 9-panel single-model detail (one CSV):
    uv run python scripts/plot_sweep.py models/hybrid_cv01_s5775_play_results.csv
"""

import argparse
import csv
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def _load_csv(path: str) -> dict[str, list[float]]:
    """Read a play_results CSV; return {column: values} as floats (True/False → 1.0/0.0)."""
    cols: dict[str, list[float]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if val in ("True", "False"):
                    cols.setdefault(key, []).append(1.0 if val == "True" else 0.0)
                else:
                    try:
                        cols.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        cols.setdefault(key, []).append(float("nan"))
    return cols


def _plot_comparison(datasets: list[tuple[str, dict[str, list[float]]]], out_path: str) -> None:
    """6-panel 2×3 head-to-head comparison across multiple models."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.flat

    for label, cols in datasets:
        speeds = cols.get("start_speed", [])
        abs_speeds = [abs(s) for s in speeds]
        x_pos_cm = [v * 100 for v in cols.get("x_pos_final", [])]
        x_vel = cols.get("x_vel_final", [])
        settle = cols.get("t_min_settle_step", [])
        theta = cols.get("theta_final", [])
        theta_dot = cols.get("theta_dot_final", [])
        acc = cols.get("acc_final", [])

        ax[0].plot(speeds, x_pos_cm, linewidth=1.5, label=label)
        ax[1].plot(speeds, x_vel, linewidth=1.5, label=label)
        ax[2].plot(abs_speeds, settle, ".", markersize=4, alpha=0.7, label=label)
        ax[3].plot(speeds, theta, linewidth=1.5, label=label)
        ax[4].plot(speeds, theta_dot, linewidth=1.5, label=label)
        ax[5].plot(speeds, acc, linewidth=1.5, label=label)

    titles = [
        ("|x| final (cm)",          "start speed (m/s)", "cm"),
        ("x_vel final (m/s)",        "start speed (m/s)", "m/s"),
        ("settle step vs |speed|",   "|start speed| (m/s)", "step"),
        ("theta final (rad)",        "start speed (m/s)", "rad"),
        ("theta_dot final (rad/s)",  "start speed (m/s)", "rad/s"),
        ("acc final (m/s²)",         "start speed (m/s)", "m/s²"),
    ]
    for axis, (title, xlabel, ylabel) in zip(ax, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel, fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=7)

    if len(datasets) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.8)

    suptitle = ", ".join(d[0] for d in datasets)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Sweep PNG: %s", out_path)


def _plot_detail(label: str, cols: dict[str, list[float]], out_path: str) -> None:
    """9-panel single-model detail figure matching play_ppo.py's _save_sweep_png layout."""
    speeds = cols.get("start_speed", [])
    no_crash_pct = [v * 100.0 for v in cols.get("no_crash", [])]

    ep_reward = cols.get("ep_reward", [])
    ep_steps = cols.get("ep_steps", [])
    rew_per_step = [
        r / s if s and not math.isnan(s) else float("nan")
        for r, s in zip(ep_reward, ep_steps)
    ]

    energy_final = cols.get("energy_final", [])
    energy_frac = [
        e / (0.5 * sp * sp) if abs(sp) > 1e-6 else float("nan")
        for e, sp in zip(energy_final, speeds)
    ]

    settle_step = cols.get("t_min_settle_step", [])
    x_pos_m = cols.get("x_pos_final", [])
    x_vel_f = cols.get("x_vel_final", [])
    x_acc_f = cols.get("acc_final", [])
    theta_f = cols.get("theta_final", [])
    thdot_f = cols.get("theta_dot_final", [])

    panels: list[tuple[str, list[float]]] = [
        ("nocrash% ↑",    no_crash_pct),
        ("rew/step ↑",    rew_per_step),
        ("energy_frac ↓", energy_frac),
        ("settle_step ↓", settle_step),
        ("x_pos_m ↓",     x_pos_m),
        ("x_vel_f",       x_vel_f),
        ("x_acc_f",       x_acc_f),
        ("theta_f",       theta_f),
        ("thdot_f ↓",     thdot_f),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(label, fontsize=12)

    for ax, (title, ys) in zip(axes.flat, panels):
        ax.plot(speeds, ys, linewidth=1.2, marker=".", markersize=2, alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("start speed (m/s)", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Detail PNG: %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Plot speed-sweep results from *_play_results.csv files.")
    parser.add_argument("csvs", nargs="+", help="One or more play_results CSV paths")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    datasets: list[tuple[str, dict[str, list[float]]]] = []
    for csv_path in args.csvs:
        stem = Path(csv_path).stem
        label = stem[:-13] if stem.endswith("_play_results") else stem
        cols = _load_csv(csv_path)
        datasets.append((label, cols))
        LOGGER.info("Loaded %s  (%d episodes)", label, len(cols.get("start_speed", [])))

    if len(datasets) == 1:
        label, cols = datasets[0]
        if args.output:
            out_path = args.output
        else:
            stem = Path(args.csvs[0]).stem
            if stem.endswith("_play_results"):
                stem = stem[:-13]
            out_path = str(Path(args.csvs[0]).parent / f"{stem}_detail.png")
        _plot_detail(label, cols, out_path)
    else:
        if args.output:
            out_path = args.output
        else:
            first_stem = Path(args.csvs[0]).stem
            if first_stem.endswith("_play_results"):
                first_stem = first_stem[:-13]
            out_path = str(Path(args.csvs[0]).parent / f"{first_stem}_sweep.png")
        _plot_comparison(datasets, out_path)


if __name__ == "__main__":
    main()
