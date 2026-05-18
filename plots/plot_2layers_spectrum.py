#!/usr/bin/env python3
"""Multi-panel NeurIPS figure: true 2-layer spectrum for several alpha values.

Each panel shows the L1_true bulk histogram (bar + step outline) with the
L2_true eigenvalues marked as downward triangles (signal spikes), averaged
over seeds when multiple seeds are available.

Usage:
  python plots/plot_2layers_spectrum.py --d 140 --gamma 0.4 --g_name id \\
      --alphas 1.5 2.0 2.5 3.0 3.5 --draft
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
from matplotlib.transforms import blended_transform_factory
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _plot_utils import NeurIPSFigure  # noqa: E402

DEFAULT_ALPHAS = [1.5, 2.0, 2.5, 3.0, 3.5]


def _results_dir(root: Path, d: int, alpha: float, gamma: float, g_name: str) -> Path:
    tag = f"d{d}_a{alpha:.1f}_g{gamma:.2f}_{g_name}"
    return root / "results" / f"2layers_spectrum_{tag}"


def load_spectrum(results_dir: Path) -> dict[str, np.ndarray] | None:
    npz = results_dir / "eigs_all_seeds.npz"
    if not npz.exists():
        return None
    f = np.load(npz)
    return {k: f[k] for k in f.files}


def plot_panel(ax: plt.Axes, data: dict[str, np.ndarray] | None,
               alpha: float, p: int) -> None:
    usetex = plt.rcParams.get("text.usetex", False)
    alpha_str = rf"$\alpha={alpha:.1f}$" if usetex else f"a={alpha:.1f}"
    ax.set_title(alpha_str)
    ax.set_xlabel("Eigenvalue")

    if data is None:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="0.6", style="italic")
        return

    # Mean over seeds
    l1 = data["L1_true"].mean(axis=0)   # (m,)

    # Bulk histogram
    n_bins = min(150, max(60, len(l1) // 50))
    hist_vals, bin_edges = np.histogram(l1, bins=n_bins, density=True)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    nz = hist_vals > 0

    ax.bar(bin_edges[:-1][nz], hist_vals[nz], width=bin_widths[nz],
           align="edge", color="#B5E4EA", alpha=1.0, linewidth=0, zorder=1)
    ax.hist(l1, bins=bin_edges, density=True, histtype="step",
            color="#8EBDCE", linewidth=0.8, zorder=2)
    ax.set_yscale("log")

    # Spike triangles: top-p eigenvalues of L1_true by |λ|
    spike_idx = np.argsort(np.abs(l1))[::-1][:p]
    spikes = l1[spike_idx]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(spikes, np.full(len(spikes), 0.07),
            transform=trans, marker="v", linestyle="none",
            markersize=5, markerfacecolor="#FFAAAA",
            markeredgecolor="#A52A2A", markeredgewidth=0.7, zorder=5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d",       type=int,   default=140)
    ap.add_argument("--gamma",   type=float, default=0.4)
    ap.add_argument("--g_name",  type=str,   default="id")
    ap.add_argument("--p_exp",   type=float, default=0.5,
                    help="p = round(d^p_exp) — used to infer #spikes")
    ap.add_argument("--alphas",  type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--draft",   action="store_true")
    ap.add_argument("--out",     type=str, default=None)
    args = ap.parse_args()

    d = args.d
    p = max(1, round(d ** args.p_exp))
    alphas = sorted(args.alphas)
    ncols = len(alphas)

    out_name = f"2layers_spectrum_d{d}_g{args.gamma:.2f}_{args.g_name}.pdf"
    out = Path(args.out) if args.out else ROOT / "figures" / out_name

    with NeurIPSFigure(
        width=1.0,
        ncols=ncols,
        nrows=1,
        aspect=1.2,
        draft=args.draft,
        save=True,
        out_path=out,
    ) as (fig, axes):
        axes_flat = np.atleast_1d(axes)
        for ax, alpha in zip(axes_flat, alphas):
            rdir = _results_dir(ROOT, d, alpha, args.gamma, args.g_name)
            data = load_spectrum(rdir)
            plot_panel(ax, data, alpha, p)

        axes_flat[0].set_ylabel("Density")

        # Shared legend on last panel
        bulk_patch = mpatches.Patch(color="#B5E4EA", label="Bulk")
        spike_handle = Line2D([0], [0], marker="v", linestyle="none",
                              markerfacecolor="#FFAAAA", markeredgecolor="#A52A2A",
                              markeredgewidth=0.7, markersize=5,
                              label=rf"Top $p={p}$ spikes")
        axes_flat[-1].legend(handles=[bulk_patch, spike_handle], loc="upper right")


if __name__ == "__main__":
    main()
