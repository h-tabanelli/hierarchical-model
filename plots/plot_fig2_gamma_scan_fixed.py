#!/usr/bin/env python3
"""Paper figure 2: fixed d=400, eps=0.5, varying gamma.

Two-panel figure:
(A) MSE vs alpha for several gamma
(B) overlap on h^(1) vs alpha for several gamma

Uses the NeurIPS plotting utilities/style from the repo.
Run with --draft on cluster nodes without LaTeX.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _plot_utils import NeurIPSFigure  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPS = [
    ("D400_eps05_g00_v2", 0.0),
    ("D400_eps05_g02_v2", 0.2),
    ("D400_eps05_g04_v2", 0.4),
    ("D400_eps05_g06_v2", 0.6),
    ("D400_eps05_g08_v2", 0.8),
    ("D400_eps05_g10_v2", 1.0),
]

RESULTS_ROOT = ROOT / "results"
OUT_PATH = ROOT / "figures" / "paper_fig2_gamma_scan.pdf"

METRIC_COL = "mse"
OVERLAP_COL = "ovH"

COLORS = {
    0.0: "C0",
    0.2: "C1",
    0.4: "C2",
    0.6: "C3",
    0.8: "C4",
    1.0: "C5",
}

BAND_ALPHA = 0.15
LINEWIDTH = 2.0
MARKERSIZE = 4.5


def _label(letter: str, subtitle: str = "") -> str:
    usetex = plt.rcParams.get("text.usetex", False)
    tag = rf"\textbf{{({letter})}}" if usetex else f"({letter})"
    return (tag + " " + subtitle).strip() if subtitle else tag


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_metrics_for_exp(exp_id: str) -> pd.DataFrame:
    base = RESULTS_ROOT / exp_id
    rows: list[dict] = []
    for fp in base.rglob("metrics.jsonl"):
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "alpha" in df.columns:
        df["alpha"] = df["alpha"].astype(float)
    return df


def aggregate_curve(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    stats = (
        df.groupby("alpha")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("alpha")
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    return stats


def _no_data_msg(ax: plt.Axes) -> None:
    ax.text(
        0.5, 0.5, "no data yet",
        transform=ax.transAxes, ha="center", va="center",
        color="0.6", style="italic",
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(ax: plt.Axes, curves: dict[float, pd.DataFrame], value_col: str) -> None:
    any_data = False
    for gamma, stats in curves.items():
        if stats.empty:
            continue
        any_data = True
        x = stats["alpha"].to_numpy()
        y = stats["mean"].to_numpy()
        sem = np.nan_to_num(stats["sem"].to_numpy(), nan=0.0)
        color = COLORS[gamma]
        ax.plot(
            x, y,
            color=color,
            lw=LINEWIDTH,
            marker="o",
            ms=MARKERSIZE,
            label=rf"$\gamma={gamma:.1f}$",
        )
        lo = y - sem
        hi = y + sem
        if value_col == OVERLAP_COL:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        else:
            lo = np.maximum(lo, 1e-12)
            hi = np.maximum(hi, lo * (1 + 1e-12))
        ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0)

    if not any_data:
        _no_data_msg(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true", help="Disable LaTeX rendering")
    ap.add_argument("--out", type=str, default=None, help="Override output PDF path")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH

    mse_curves: dict[float, pd.DataFrame] = {}
    ov_curves: dict[float, pd.DataFrame] = {}

    for exp_id, gamma in EXPS:
        df = load_metrics_for_exp(exp_id)
        if df.empty:
            mse_curves[gamma] = pd.DataFrame()
            ov_curves[gamma] = pd.DataFrame()
            continue

        if METRIC_COL not in df.columns or OVERLAP_COL not in df.columns:
            mse_curves[gamma] = pd.DataFrame()
            ov_curves[gamma] = pd.DataFrame()
            continue

        mse_curves[gamma] = aggregate_curve(df, METRIC_COL)
        ov_curves[gamma] = aggregate_curve(df, OVERLAP_COL)

    # Full text width, two panels, close to reference proportions.
    with NeurIPSFigure(
        width=1.0,
        ncols=2,
        nrows=1,
        aspect=0.42,
        draft=args.draft,
        save=True,
        out_path=out,
    ) as (fig, axes):
        ax1, ax2 = axes

        plot_panel(ax1, mse_curves, METRIC_COL)
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel("MSE")
        ax1.set_title(_label("A"))
        ax1.set_xlim(0.95, 3.45)
        ax1.legend(frameon=False, ncol=2, loc="lower left")

        plot_panel(ax2, ov_curves, OVERLAP_COL)
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"Feature overlap on $h^{(1)}$")
        ax2.set_ylim(0.0, 1.02)
        ax2.set_xlim(0.95, 3.45)
        ax2.set_title(_label("B"))

        # Shared figure title kept compact and away from the axes.
        fig.suptitle(r"Fixed $d=400$, $\epsilon=0.5$, varying $\gamma$", y=1.02)


if __name__ == "__main__":
    main()
