#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True, help="Path to rf1_spectrum_alpha=....pt")
    ap.add_argument("--out", type=str, default=None, help="Explicit output path (.pdf). Mutually exclusive with --figroot.")
    ap.add_argument("--figroot", type=str, default=None,
                    help="Root dir for structured output: figroot/d={d}/g={g}/normalize_w={0|1}/alpha={a}/spectrum.pdf")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--topk_spikes", type=int, default=20)
    ap.add_argument("--topk_abs", type=int, default=100)
    args = ap.parse_args()

    if args.out is None and args.figroot is None:
        ap.error("Provide either --out or --figroot")

    obj = torch.load(args.pt, map_location="cpu")
    metrics = obj["metrics"]

    # resolve output path
    if args.figroot is not None:
        g = metrics.get("g_name", "id")
        normw = int(metrics.get("normalize_w", False))
        fresh = int(metrics.get("fresh_proj", False))
        subdir = (f"d={metrics['d']}/g={g}/"
                  f"normw={normw}_fresh={fresh}/"
                  f"rfw={metrics['rf_width']}/alpha={metrics['alpha']:.4f}")
        out = (Path(args.figroot) / subdir / "spectrum").with_suffix(".pdf")
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    full_eigs = obj["full_eigs"].numpy()
    abs_full_eigs = np.sort(np.abs(full_eigs))[::-1]

    s_mean = obj["s_mean"].numpy()
    s_var  = obj["s_var"].numpy()
    post_proj_mean = obj["post_proj_mean"].numpy()
    post_proj_var  = obj["post_proj_var"].numpy()

    # --- shared style ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.5,
    })

    # =========================================================
    # Figure 1: spectrum (log-scale density + spikes from full_eigs)
    # =========================================================
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Histogram via numpy so we can filter zero bins for log scale
    n_bins = min(150, max(60, len(full_eigs) // 2))
    hist_vals, bin_edges = np.histogram(full_eigs, bins=n_bins, density=True)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    nonzero = hist_vals > 0
    ax.bar(
        bin_edges[:-1][nonzero],
        hist_vals[nonzero],
        width=bin_widths[nonzero],
        align="edge",
        color="#ef8f8f",
        edgecolor="#9c9c9c",
        linewidth=0.6,
        alpha=0.85,
        label="Bulk (noise)",
        zorder=1,
    )
    ax.set_yscale("log")

    # Spikes: top-p eigenvalues by |λ| from full diagonalization
    p_true = int(metrics.get("p", args.topk_spikes))
    topk = min(args.topk_spikes, p_true)
    spike_indices = np.argsort(np.abs(full_eigs))[::-1][:topk]
    spike_eigs = full_eigs[spike_indices]

    for i, ev in enumerate(spike_eigs):
        ax.axvline(
            ev,
            color="red",
            alpha=0.65,
            linewidth=1.4,
            zorder=4,
            label="Spikes (full diag)" if i == 0 else None,
        )

    # Title / labels
    ttl = args.title
    if ttl is None:
        ttl = (
            f"RF1 spectrum — d={metrics['d']}, α={metrics['alpha']}, "
            f"p₁={metrics['rf_width']}"
        )
    ax.set_title(ttl, pad=10)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density (log scale)")

    x_pad = 0.04 * (full_eigs.max() - full_eigs.min() + 1e-12)
    ax.set_xlim(full_eigs.min() - x_pad, full_eigs.max() + x_pad)

    bulk_handle = Patch(facecolor="#ef8f8f", edgecolor="#9c9c9c", linewidth=0.6, label="Bulk (noise)")
    spike_handle = Line2D([0], [0], color="red", linestyle="-", linewidth=1.4, alpha=0.65, label="Spikes (full diag)")
    leg = ax.legend(handles=[bulk_handle, spike_handle], loc="upper right", frameon=True,
                    framealpha=0.8)
    leg.get_frame().set_edgecolor("#b0b0b0")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.png')}")

    # =========================================================
    # Figure 2: top |eigenvalues| vs rank
    # =========================================================
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    topk_abs = min(args.topk_abs, len(abs_full_eigs))
    ranks = np.arange(1, topk_abs + 1)

    ax.plot(ranks, abs_full_eigs[:topk_abs], color="k", marker=".", markersize=0.5,
            linewidth=1.8, label=r"$|\lambda_i|$ (full spectrum)")

    d1 = int(metrics["p"])
    cutoff = d1 + 0.5
    ax.axvspan(0.5, cutoff, color="red", alpha=0.06, zorder=0)
    ax.axvline(cutoff, linestyle="--", linewidth=1.2, color="black", label=r"cutoff at $d_1$")

    ttl2 = args.title
    if ttl2 is None:
        ttl2 = (
            r"$|\lambda|$ vs rank — "
            f"d={metrics['d']}, α={metrics['alpha']}, p₁={metrics['rf_width']}"
        )
    ax.set_title(ttl2, pad=10)
    ax.set_xlabel("Rank")
    ax.set_ylabel(r"$|\lambda|$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    out2_pdf = out.with_name(out.stem + "_topabs.pdf")
    out2_png = out.with_name(out.stem + "_topabs.png")

    fig.tight_layout()
    fig.savefig(out2_pdf, bbox_inches="tight")
    fig.savefig(out2_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2_pdf}")
    print(f"Saved {out2_png}")

    # =========================================================
    # Figure 3: mean / variance vs index (pre vs post-projection)
    # =========================================================
    fig3, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    x_pre  = np.linspace(0.0, 1.0, len(s_var))
    x_post = np.linspace(0.0, 1.0, len(post_proj_var))

    ttl3 = args.title
    if ttl3 is None:
        ttl3 = (f"d={metrics['d']}, α={metrics['alpha']}, "
                f"p₁={metrics['rf_width']}, d₁={metrics['p']}")
    fig3.suptitle(ttl3, y=0.99)

    axes[0].plot(x_pre,  s_var,         "o--", linewidth=1.4, markersize=3, label="pre")
    axes[0].plot(x_post, post_proj_var, "s-",  linewidth=1.4, markersize=3, label="post")
    axes[0].set_title("variance")
    axes[0].set_xlabel("index (normalized)")
    axes[0].set_ylabel("var")
    axes[0].legend(frameon=False)

    axes[1].plot(x_pre,  s_mean,         "o--", linewidth=1.4, markersize=3, label="pre")
    axes[1].plot(x_post, post_proj_mean, "s-",  linewidth=1.4, markersize=3, label="post")
    axes[1].set_title("mean")
    axes[1].set_xlabel("index (normalized)")
    axes[1].set_ylabel("mean")
    axes[1].legend(frameon=False)

    out_stats = out.with_name(out.stem + "_projstats" + out.suffix)
    out_stats_png = out_stats.with_suffix(".png")

    fig3.tight_layout()
    fig3.savefig(out_stats, dpi=200, bbox_inches="tight")
    fig3.savefig(out_stats_png, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {out_stats}")
    print(f"Saved {out_stats_png}")


if __name__ == "__main__":
    main()
