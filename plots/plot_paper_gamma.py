#!/usr/bin/env python3
"""2-panel paper figure: (A) nMSE vs alpha, (B) overlap H1 vs alpha for several gamma values.

Data source: results/D400_eps05_g*_v2/**/metrics.jsonl  (d=400, eps=0.5, g_name=id).
Each gamma value gives one curve; missing/incomplete data is handled gracefully.

Figure width: 1.0 x NeurIPS text width (5.5 in).  Rendered with LaTeX via neurips.mplstyle.
Run with --draft to skip LaTeX (faster iteration).
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

RESULTS_GLOB = str(ROOT / "results" / "D400_eps05_g*_v2" / "**" / "metrics.jsonl")
OUT_PATH = ROOT / "figures" / "paper_gamma.pdf"

GAMMA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_cmap = plt.colormaps["plasma"]
_COLORS = [_cmap(v) for v in np.linspace(0.15, 0.82, len(GAMMA_VALUES))]
GAMMA_COLOR = dict(zip(GAMMA_VALUES, _COLORS))


def _label(letter: str) -> str:
    usetex = plt.rcParams.get("text.usetex", False)
    return rf"\textbf{{({letter})}}" if usetex else f"({letter})"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for fp in glob.glob(RESULTS_GLOB, recursive=True):
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
        return pd.DataFrame(columns=["alpha", "gamma", "nmse", "ovH"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _no_data_msg(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "no data yet",
            transform=ax.transAxes, ha="center", va="center",
            color="0.6", style="italic")


def _plot_mean_sem(ax: plt.Axes, df: pd.DataFrame, col: str,
                   gamma_val: float, color: str, label: str) -> None:
    grp = df[np.isclose(df["gamma"], gamma_val)].sort_values("alpha")
    if grp.empty:
        return
    stats = grp.groupby("alpha")[col].agg(["mean", "std", "count"]).reset_index()
    stats = stats.sort_values("alpha")
    sem = stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats["alpha"], stats["mean"], color=color, label=label)
    ax.fill_between(stats["alpha"],
                    stats["mean"] - sem,
                    stats["mean"] + sem,
                    color=color, alpha=0.15, linewidth=0)


def _gamma_label(g: float) -> str:
    usetex = plt.rcParams.get("text.usetex", False)
    if usetex:
        return rf"$\gamma={g:.1f}$"
    return rf"$\gamma={g:.1f}$"


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "nmse" not in df.columns:
        _no_data_msg(ax)
    else:
        for g in GAMMA_VALUES:
            _plot_mean_sem(ax, df, "nmse", g, GAMMA_COLOR[g], _gamma_label(g))
    ax.set_xlabel(r"$\alpha$")
    ax.set_yscale("log")
    ax.set_ylabel(r"MSE")
    ax.set_xlim(1.5, 3.4)
    # ax.set_title(_label("A"))


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "ovH" not in df.columns:
        _no_data_msg(ax)
    else:
        for g in GAMMA_VALUES:
            _plot_mean_sem(ax, df, "ovH", g, GAMMA_COLOR[g], _gamma_label(g))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Overlap $h^{(1)}$")
    ax.set_xlim(1.5, 3.4)
    # ax.set_title(_label("B"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true", help="Disable LaTeX rendering")
    ap.add_argument("--out", type=str, default=None, help="Override output PDF path")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH
    df = load_metrics()

    with NeurIPSFigure(
        width=1.0,
        ncols=2,
        nrows=1,
        aspect=0.5,
        draft=args.draft,
        save=True,
        out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], df)
        plot_panel_b(axes[1], df)


if __name__ == "__main__":
    main()
