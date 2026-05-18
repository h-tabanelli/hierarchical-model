#!/usr/bin/env python3
"""2-panel paper figure for the power-law teacher (no random features).

(A) MSE vs alpha for several d values.
(B) Overlap H1 vs alpha for several d values.

Data: gamma=0.4, g*=id, B_mode=powerlaw_diag, model=true.
  d=120 : results/D120_eps05_g04          (alpha 1.0-3.8,  10 seeds)
  d=250 : results/D250_eps05_g04          (alpha 2.0-3.2,  10 seeds)
  d=400 : results/D400_eps05_g04_v2       (alpha 1.5-3.4,  10 seeds)

Run with --draft to skip LaTeX.
"""
from __future__ import annotations

import argparse
import glob
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

SOURCES = {
    120: ROOT / "results" / "D120_eps05_g04_tanh_cv_deg3_fr5_pl4",
    250: ROOT / "results" / "D250_eps05_g04_tanh_cv_deg3_fr5_pl4",
    400: ROOT / "results" / "D400_eps05_g04_tanh_cv_deg3_fr5_pl4",
}
D_VALUES = [120, 250, 400]
_COLORS = [(255/255, 63/255, 69/255), (255/255, 212/255, 41/255), (95/255, 144/255, 255/255)]
D_COLOR = dict(zip(D_VALUES, _COLORS))

OUT_PATH = ROOT / "figures" / "paper_powerlaw_tanh.pdf"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for d_val, src_dir in SOURCES.items():
        for fp in glob.glob(str(src_dir / "**" / "metrics.jsonl"), recursive=True):
            with open(fp) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not rows:
        return pd.DataFrame(columns=["alpha", "d", "nmse", "ovH"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _no_data_msg(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
            ha="center", va="center", color="0.6", style="italic")


def _plot_mean_sem(ax: plt.Axes, df: pd.DataFrame, col: str, d_val: int) -> None:
    grp = df[df["d"] == d_val].sort_values("alpha")
    if grp.empty:
        return
    stats = grp.groupby("alpha")[col].agg(["mean", "std", "count"]).reset_index()
    sem = stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats["alpha"], stats["mean"], color=D_COLOR[d_val], label=rf"$d={d_val}$")
    ax.fill_between(stats["alpha"], stats["mean"] - sem, stats["mean"] + sem,
                    color=D_COLOR[d_val], alpha=0.15, linewidth=0)


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "nmse" not in df.columns:
        _no_data_msg(ax)
    else:
        for d_val in D_VALUES:
            _plot_mean_sem(ax, df, "nmse", d_val)
        ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"MSE")
    ax.set_yscale("log")
    #ax.set_ylim(1e-1, 1.1)
    ax.set_xlim(1.5, 3.4)



def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "ovH" not in df.columns:
        _no_data_msg(ax)
    else:
        for d_val in D_VALUES:
            _plot_mean_sem(ax, df, "ovH", d_val)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Overlap $h^{(1)}$")
    ax.set_xlim(1.5, 3.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out",   type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH
    df  = load_metrics()

    with NeurIPSFigure(
        width=1.0, ncols=2, nrows=1, aspect=0.5,
        draft=args.draft, save=True, out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], df)
        plot_panel_b(axes[1], df)


if __name__ == "__main__":
    main()
