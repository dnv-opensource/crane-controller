"""Plot training curves from one or more *_log.csv files produced by EpRewardLogCallback.

Examples:
--------
.. code-block:: bash

    uv run python scripts/plot_training.py models/hybrid_cv01_s5775_log.csv
    uv run python scripts/plot_training.py models/hybrid_cv01_s5775_log.csv models/sig_t_min_s5775_log.csv
    uv run python scripts/plot_training.py models/hybrid_cv01_s5775_log.csv --output comparison.png
"""

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOGGER = logging.getLogger(__name__)

PANELS: list[tuple[str, str]] = [
    ("rail_hit%↓", "rail_hit_pct"),
    ("rew/step↑", "rew_per_step"),
    ("ep_len↑", "ep_len_mean"),
    ("expl_var↑", "explained_variance"),
    ("value_loss↓", "value_loss"),
    ("entropy~", "entropy_loss"),
    ("approx_kl↓", "approx_kl"),
    ("clip_frac↓", "clip_fraction"),
    ("|θ̇|↓", "mean_theta_dot_abs"),
    ("|x|↓ (m)", "mean_x_pos_abs"),
    ("|xv|↓", "mean_x_vel_abs"),
    ("energy↓", "mean_energy"),
]


def _load_csv(path: str) -> tuple[list[float], dict[str, list[float]]]:
    """Read a training log CSV; return (timesteps, {column: values}) with NaN for missing entries."""
    ts: list[float] = []
    cols: dict[str, list[float]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["t"]))
            except (ValueError, KeyError):
                continue
            for key, val in row.items():
                if key == "t":
                    continue
                try:
                    cols.setdefault(key, []).append(float(val))
                except ValueError:
                    cols.setdefault(key, []).append(float("nan"))
    return ts, cols


def main() -> None:
    """Parse CLI arguments and produce the training curve PNG."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Plot training curves from *_log.csv files.")
    parser.add_argument("csvs", nargs="+", help="One or more training log CSV paths")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: first CSV stem + _training.png in the same directory)",
    )
    args = parser.parse_args()

    datasets: list[tuple[str, list[float], dict[str, list[float]]]] = []
    for csv_path in args.csvs:
        stem = Path(csv_path).stem
        label = stem.removesuffix("_log")
        ts, cols = _load_csv(csv_path)
        datasets.append((label, ts, cols))
        LOGGER.info("Loaded %s  (%d rows)", label, len(ts))

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")

    linestyles = ["-", "--", "-.", ":"]
    for ax, (title, col_key) in zip(axes.flat, PANELS, strict=True):
        for i, (label, ts, cols) in enumerate(datasets):
            ys = cols.get(col_key, [float("nan")] * len(ts))
            ax.plot(ts, ys, linewidth=1.5, label=label, linestyle=linestyles[i % len(linestyles)])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("steps", fontsize=8)
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(labelsize=7)
        ax.grid(visible=True, alpha=0.3)

    if len(datasets) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.8)

    suptitle = ", ".join(d[0] for d in datasets)
    fig.suptitle(suptitle, fontsize=11)  # pyright: ignore[reportAttributeAccessIssue]
    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        first_stem = Path(args.csvs[0]).stem.removesuffix("_log")
        out_path = str(Path(args.csvs[0]).parent / f"{first_stem}_training.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Training PNG: %s", out_path)


if __name__ == "__main__":
    main()
