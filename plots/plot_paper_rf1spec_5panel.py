#!/usr/bin/env python3
"""5-panel RF1 spectrum figure (d=100, g*=tanh, alpha=1.5..3.5).

Each panel shows the eigenvalue density histogram (log scale) with the top-d1
spikes marked as downward triangles, matching the style of plot_paper_abc.py.

Usage:
    python plots/plot_paper_rf1spec_5panel.py [--draft] [--out figures/...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _plot_utils import setup_style, save_or_show  # noqa: E402

ALPHAS = [1.5, 2.0, 2.5, 3.0, 3.5]
SPEC_DIR = ROOT / "results" / "paper_rf1spec_d100_gtanh" / "chunk=0000" / "seed=0000"
OUT_PATH = ROOT / "figures" / "paper_rf1spec_5panel.pdf"


def _pt_path(alpha: float) -> Path:
    return SPEC_DIR / f"rf1_spectrum_alpha={alpha:.4f}.pt"


def _load(alpha: float) -> tuple[np.ndarray, int]:
    obj = torch.load(str(_pt_path(alpha)), map_location="cpu")
    return obj["full_eigs"].numpy(), int(obj["metrics"]["p"])


def _plot_panel(ax: plt.Axes, eigs: np.ndarray, d1: int, alpha: float) -> None:
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

    spike_idx = np.argsort(np.abs(eigs))[::-1][:d1]
    spike_eigs = eigs[spike_idx]
    # y position: density at each spike's bin so the marker tip sits on the bar top
    bin_indices = np.clip(np.searchsorted(bin_edges[1:], spike_eigs), 0, len(hist_vals) - 1)
    spike_densities = hist_vals[bin_indices]
    valid = spike_densities > 0
    ax.plot(
        spike_eigs[valid], spike_densities[valid],
        marker="v", linestyle="none",
        markersize=4, markerfacecolor="#FFAAAA",
        markeredgecolor="#A52A2A", markeredgewidth=0.6, zorder=5,
    )

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, symmetric=True))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title(rf"$\alpha={alpha:.1f}$")
    ax.set_xlabel(r"Eigenvalue")
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true", help="Disable LaTeX rendering")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH

    setup_style(draft=args.draft)
    plt.rcParams["figure.constrained_layout.use"] = False

    # Geometry: full NeurIPS text width (5.5 in) for the 5 panels, plus a
    # left strip that hosts the legend outside all axes.
    TEXTWIDTH = 5.5
    ncols = len(ALPHAS)
    aspect = 1.3
    h_total = TEXTWIDTH * aspect / ncols

    fig, axes = plt.subplots(1, ncols, figsize=(TEXTWIDTH, h_total), sharey=True)

    data = [_load(a) for a in ALPHAS]

    for ax, (eigs, d1), alpha in zip(axes, data, ALPHAS):
        _plot_panel(ax, eigs, d1, alpha)

    axes[0].set_ylabel(r"Density")

    fig.tight_layout(w_pad=0.3)

    save_or_show(fig, True, out, also_png=True)


if __name__ == "__main__":
    main()

