#!/usr/bin/env python3
"""3-panel paper figure: (A) nMSE vs alpha, (B) overlap H1 vs alpha, (C) RF1 spectrum.

Panels A and B load metrics from results/results 2/**/metrics.jsonl for g_name='tanh'.
Panel C loads the RF1 eigenspectrum at alpha=3 from paper_rf1spec_d100_gtanh.

Figure width: 1.0 x NeurIPS text width (5.5 in).  Rendered with LaTeX via neurips.mplstyle.
Run with --draft to skip LaTeX (faster, no TeX fonts).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _plot_utils import NeurIPSFigure  # noqa: E402

RESULTS2_GLOBS = [
    str(ROOT / "results" / "rf2cw_metrics_only_v2" / "results" / "rf2cw_d80_tanh"  / "**" / "metrics.jsonl"),
    str(ROOT / "results" / "rf2cw_metrics_only_v2" / "results" / "rf2cw_d100_tanh" / "**" / "metrics.jsonl"),
    str(ROOT / "results" / "rf2cw_d120_tanh" / "**" / "metrics.jsonl"),
    str(ROOT / "results" / "rf2cw_d140_tanh" / "**" / "metrics.jsonl"),
]
SPECTRUM_PT = (
    ROOT / "results" / "paper_rf1spec_d100_gtanh"
    / "chunk=0000" / "seed=0000" / "rf1_spectrum_alpha=3.0000.pt"
)
OUT_PATH = ROOT / "figures" / "paper_abc.pdf"

D_VALUES = [80, 100, 120, 140]
# Picked from matplotlib's default C0-C3 cycle so they match other figures.
_COLORS = ["C0", "C1", "C2", "C3"]
D_COLOR = dict(zip(D_VALUES, _COLORS))


def _label(letter: str, subtitle: str = "") -> str:
    """Return a panel label like (A) or \\textbf{(A)}, depending on usetex."""
    usetex = plt.rcParams.get("text.usetex", False)
    tag = rf"\textbf{{({letter})}}" if usetex else f"({letter})"
    return (tag + " " + subtitle).strip() if subtitle else tag


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for pattern in RESULTS2_GLOBS:
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
        return pd.DataFrame(columns=["alpha", "d", "g_name", "nmse", "ovH"])
    return pd.DataFrame(rows)


def _tanh_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "g_name" not in df.columns:
        return pd.DataFrame()
    return df[df["g_name"] == "tanh"].copy()


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _no_data_msg(ax: plt.Axes) -> None:
    ax.text(
        0.5, 0.5, "no data yet",
        transform=ax.transAxes, ha="center", va="center",
        color="0.6", style="italic",
    )


def _plot_mean_std(ax, df, col, d_val, color):
    grp = df[df["d"] == d_val].sort_values("alpha")
    if grp.empty:
        return
    stats = grp.groupby("alpha")[col].agg(["mean", "std", "count"]).reset_index()
    stats = stats.sort_values("alpha")
    sem = stats["std"] / np.sqrt(stats["count"])
    ax.plot(stats["alpha"], stats["mean"], color=color, label=rf"$d={d_val}$")
    ax.fill_between(
        stats["alpha"],
        stats["mean"] - sem,
        stats["mean"] + sem,
        color=color, alpha=0.15, linewidth=0,
    )


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = _tanh_subset(df)
    if sub.empty or "nmse" not in sub.columns:
        _no_data_msg(ax)
    else:
        for d_val in D_VALUES:
            _plot_mean_std(ax, sub, "nmse", d_val, D_COLOR[d_val])
        ax.legend()
    #ax.axvline(2.5, color="k", ls="--", lw=0.8, label=r"$\alpha=k+\epsilon$")
    ax.set_xlabel(r"$\log(n)/\log(d)$")
    ax.set_ylabel(r"MSE")
    ax.set_ylim(0.55, 1.1)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    #ax.set_title(_label("A"))


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = _tanh_subset(df)
    if sub.empty or "ovH" not in sub.columns:
        _no_data_msg(ax)
    else:
        for d_val in D_VALUES:
            _plot_mean_std(ax, sub, "ovH", d_val, D_COLOR[d_val])
        #ax.legend()
    ax.set_xlabel(r"$\log(n)/\log(d)$")
    ax.set_ylabel(r"Overlap $h^{(1)}$")
    #ax.axvline(2.5, color="k", ls="--", lw=0.8, label=r"$\alpha=k+\epsilon$")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    #ax.set_title(_label("B"))


def plot_panel_c(ax: plt.Axes, pt_path: Path) -> None:
    obj = torch.load(str(pt_path), map_location="cpu")
    m = obj["metrics"]
    eigs = obj["full_eigs"].numpy()
    d1 = int(m["p"])
    alpha_val = float(m["alpha"])

    # Bulk histogram
    n_bins = min(150, max(60, len(eigs) // 50))
    hist_vals, bin_edges = np.histogram(eigs, bins=n_bins, density=True)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    nz = hist_vals > 0
    ax.bar(
        bin_edges[:-1][nz],
        hist_vals[nz],
        width=bin_widths[nz],
        align="edge",
        color="#B5E4EA",
        alpha=1,
        linewidth=0,
        zorder=1,
    )
    ax.hist(eigs, bins=bin_edges, density=True, histtype="step",
            color="#8EBDCE", linewidth=0.8, zorder=2)
    ax.set_yscale("log")

    # Top d1 eigenvalues — downward triangles at top of panel (blended transform:
    # x in data coords, y in axes coords so they sit at a fixed height regardless of ylim)
    spike_idx = np.argsort(np.abs(eigs))[::-1][:d1]
    spike_eigs = eigs[spike_idx]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        spike_eigs, np.full(len(spike_eigs), 0.07),
        transform=trans,
        marker="v", linestyle="none",
        markersize=5, markerfacecolor="#FFAAAA",
        markeredgecolor="#A52A2A", markeredgewidth=0.7, zorder=5,
    )

    ax.set_xlabel(r"Eigenvalue")
    ax.set_ylabel(r"Density")
    #ax.set_title(_label("C"))

    bulk_patch = mpatches.Patch(color="#B5E4EA", label="Bulk")
    spike_handle = Line2D([0], [0], marker="v", linestyle="none",
                          markerfacecolor="#FFAAAA", markeredgecolor="#A52A2A",
                          markeredgewidth=0.7, markersize=5,
                          label=rf"Top $d_1 = {d1}$")
    ax.legend(handles=[bulk_patch, spike_handle], loc="upper right")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)


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

    # aspect=1.2 gives each panel h/w ~ 2.2/1.83, matching the default column shape
    with NeurIPSFigure(
        width=1.0,
        ncols=3,
        nrows=1,
        aspect=1.2,
        draft=args.draft,
        save=True,
        out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], df)
        plot_panel_b(axes[1], df)
        plot_panel_c(axes[2], SPECTRUM_PT)


if __name__ == "__main__":
    main()
