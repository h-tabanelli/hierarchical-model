#!/usr/bin/env python3
"""3-panel paper figure for the power-law teacher (no random features).

(A) MSE vs alpha for several d values.
(B) Overlap H1 vs alpha for several d values.
(C) True L1 eigenspectrum at a fixed (d, alpha).

Data: gamma=0.4, g*=id, B_mode=powerlaw_diag, model=true.
  d=120 : results/D120_eps05_g04          (alpha 1.0-3.8,  10 seeds)
  d=250 : results/D250_eps05_g04          (alpha 2.0-3.2,  10 seeds)
  d=400 : results/D400_eps05_g04_v2       (alpha 1.5-3.4,  10 seeds)
Spectrum: results/2layers_spectrum_d{D}_a{alpha}_g0.40_id/eigs_all_seeds.npz

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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _plot_utils import NeurIPSFigure  # noqa: E402

SOURCES = {
    120: ROOT / "results" / "D120_eps05_g04",
    250: ROOT / "results" / "D250_eps05_g04_v2",
    400: ROOT / "results" / "D400_eps05_g04_v2",
}
D_VALUES = [120, 250, 400]
_COLORS = [(255/255, 63/255, 69/255), (255/255, 212/255, 41/255), (95/255, 144/255, 255/255)]
D_COLOR = dict(zip(D_VALUES, _COLORS))

# Panel C: spectrum placeholder — switch to d=200/alpha=3.5 once that run finishes
SPECTRUM_D     = 140
SPECTRUM_ALPHA = 3.5
SPECTRUM_P_EXP = 0.5   # p = round(d^p_exp)

OUT_PATH = ROOT / "figures" / "paper_powerlaw.pdf"


def _spectrum_path(d: int, alpha: float) -> Path:
    tag = f"d{d}_a{alpha:.1f}_g0.40_id"
    return ROOT / "results" / f"2layers_spectrum_{tag}" / "eigs_all_seeds.npz"


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


def plot_panel_c(ax: plt.Axes, d: int, alpha: float) -> None:
    npz_path = _spectrum_path(d, alpha)
    if not npz_path.exists():
        _no_data_msg(ax)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        return

    f = np.load(npz_path)
    l1 = f["L1_true"].mean(axis=0)   # mean over seeds, shape (m,)
    p  = max(1, round(d ** SPECTRUM_P_EXP))

    n_bins = min(150, max(60, len(l1) // 50))
    hist_vals, bin_edges = np.histogram(l1, bins=n_bins, density=True)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    nz = hist_vals > 0

    ax.bar(bin_edges[:-1][nz], hist_vals[nz], width=bin_widths[nz],
           align="edge", color="#B5E4EA", alpha=1.0, linewidth=0, zorder=1)
    ax.hist(l1, bins=bin_edges, density=True, histtype="step",
            color="#8EBDCE", linewidth=0.8, zorder=2)
    ax.set_yscale("log")

    spike_idx = np.argsort(np.abs(l1))[::-1][:p]
    spikes = l1[spike_idx]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(spikes, np.full(len(spikes), 0.07),
            transform=trans, marker="v", linestyle="none",
            markersize=5, markerfacecolor="#FFAAAA",
            markeredgecolor="#A52A2A", markeredgewidth=0.7, zorder=5)

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")

    bulk_patch = mpatches.Patch(color="#B5E4EA", label="Bulk")
    spike_handle = Line2D([0], [0], marker="v", linestyle="none",
                          markerfacecolor="#FFAAAA", markeredgecolor="#A52A2A",
                          markeredgewidth=0.7, markersize=5,
                          label=rf"Top $d_1={p}$")
    #ax.legend(handles=[bulk_patch, spike_handle], loc="upper right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out",   type=str, default=None)
    ap.add_argument("--spec_d",     type=int,   default=SPECTRUM_D,
                    help="d for panel C spectrum")
    ap.add_argument("--spec_alpha", type=float, default=SPECTRUM_ALPHA,
                    help="alpha for panel C spectrum")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT_PATH
    df  = load_metrics()

    with NeurIPSFigure(
        width=1.0, ncols=3, nrows=1, aspect=0.8,
        draft=args.draft, save=True, out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], df)
        plot_panel_b(axes[1], df)
        plot_panel_c(axes[2], args.spec_d, args.spec_alpha)


if __name__ == "__main__":
    main()
