#!/usr/bin/env python3
"""2-panel paper figure: MSE vs alpha and overlap H1 vs alpha for several g* non-linearities.

Data: d=400, gamma=0.4, 10 seeds.
  - g*=id          : results/D400_eps05_g04_v2
  - g*=tanh        : results/D400_eps05_g04_tanh_saveest
  - g*=x2+x-1      : results/D400_eps05_g04_x2pxm1_saveest

Run with --draft to skip LaTeX rendering.
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
    "id":               ROOT / "results" / "D400_eps05_g04_v2",
    "tanh":             ROOT / "results" / "D400_eps05_g04_tanh_saveest",
    "x2_plus_x_minus1": ROOT / "results" / "D400_eps05_g04_x2pxm1_saveest",
}

OUT_PATH = ROOT / "figures" / "paper_gstar.pdf"

_cmap = plt.colormaps["Set1"]
G_COLOR = {
    "id":               _cmap(0),
    "tanh":             _cmap(1),
    "x2_plus_x_minus1": _cmap(2),
}

def _g_label(g: str) -> str:
    usetex = plt.rcParams.get("text.usetex", False)
    labels = {
        "id":               r"$g^\star = \mathrm{id}$",
        "tanh":             r"$g^\star = \tanh$",
        "x2_plus_x_minus1": r"$g^\star = x^2+x-1$",
    }
    return labels.get(g, g)


def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for g_name, src_dir in SOURCES.items():
        pattern = str(src_dir / "**" / "metrics.jsonl")
        for fp in glob.glob(pattern, recursive=True):
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
        return pd.DataFrame(columns=["alpha", "g_name", "nmse", "ovH"])
    return pd.DataFrame(rows)


def _no_data_msg(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
            ha="center", va="center", color="0.6", style="italic")


def _plot_mean_sem(ax: plt.Axes, df: pd.DataFrame, col: str, g: str) -> None:
    grp = df[df["g_name"] == g].sort_values("alpha")
    if grp.empty:
        return
    stats = grp.groupby("alpha")[col].agg(["mean", "std", "count"]).reset_index()
    sem = stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats["alpha"], stats["mean"], color=G_COLOR[g], label=_g_label(g))
    ax.fill_between(stats["alpha"], stats["mean"] - sem, stats["mean"] + sem,
                    color=G_COLOR[g], alpha=0.15, linewidth=0)


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "nmse" not in df.columns:
        _no_data_msg(ax)
    else:
        for g in SOURCES:
            _plot_mean_sem(ax, df, "nmse", g)
        ax.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"MSE")
    ax.set_yscale("log")


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty or "ovH" not in df.columns:
        _no_data_msg(ax)
    else:
        for g in SOURCES:
            _plot_mean_sem(ax, df, "ovH", g)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Overlap $h^{(1)}$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH
    df = load_metrics()

    with NeurIPSFigure(
        width=1.0, ncols=2, nrows=1, aspect=0.8,
        draft=args.draft, save=True, out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], df)
        plot_panel_b(axes[1], df)


if __name__ == "__main__":
    main()
