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
    ap.add_argument("--out", type=str, required=True, help="Output figure path (.pdf or .png)")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--topk_spikes", type=int, default=20)
    ap.add_argument("--topk_abs", type=int, default=100)
    args = ap.parse_args()

    obj = torch.load(args.pt, map_location="cpu")
    metrics = obj["metrics"]
    full_eigs = obj["full_eigs"].numpy()
    selected_ritz_eigs = obj["selected_ritz_eigs"].numpy()
    abs_full_eigs = np.sort(np.abs(full_eigs))[::-1]
    abs_ritz_eigs = np.sort(np.abs(selected_ritz_eigs))[::-1]

    s_mean = obj["s_mean"].numpy()
    s_var  = obj["s_var"].numpy()

    post_proj_mean = obj["post_proj_mean"].numpy()
    post_proj_var  = obj["post_proj_var"].numpy()

    # --- style close to your reference figure ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    #ax.set_facecolor("#ececec")

    # Bulk histogram
    n_bins = min(150, max(60, len(full_eigs) // 2))
    hist_vals, bin_edges, _ = ax.hist(
        full_eigs,
        bins=n_bins,
        density=True,
        color="#ef8f8f",
        edgecolor="#9c9c9c",
        linewidth=2.0,
        alpha=0.85,
        label="Bulk (noise)",
        zorder=1,
    )

    # Spikes as downward triangles near the x-axis
    topk = min(args.topk_spikes, len(selected_ritz_eigs))
    y_max = float(hist_vals.max()) if len(hist_vals) else 1.0
    y_spike = 0.045 * y_max

    ax.scatter(
        selected_ritz_eigs[:topk],
        np.full(topk, y_spike),
        marker="v",
        s=520,
        color="red",
        edgecolors="red",
        linewidths=1.5,
        zorder=4,
        alpha=0.5,
        label="Spikes (signal)",
    )

    # Title / labels
    ttl = args.title
    if ttl is None:
        ttl = (
            f"RF1 spectrum, d={metrics['d']}, alpha={metrics['alpha']}, "
            f"$p_1$={metrics['rf_width']}"
        )
    ax.set_title(ttl, pad=18)
    ax.set_xlabel("Eigenvalue", labelpad=14)
    ax.set_ylabel("Density", labelpad=18)
    ax.set_yscale("log")

    # Grid
    #ax.grid(True, axis="y", alpha=0.25, linewidth=1.5)
    #ax.grid(False, axis="x")

    # Limits with a bit of margin
    x_all = np.concatenate([full_eigs, selected_ritz_eigs[:topk]])
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    x_pad = 0.04 * (x_max - x_min + 1e-12)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.0, 1.08 * y_max)

    # Clean legend with custom handles
    bulk_handle = Patch(facecolor="#ef8f8f", edgecolor="#9c9c9c", linewidth=2.0, label="Bulk (noise)")
    spike_handle = Line2D(
        [0], [0],
        marker="v",
        color="red",
        linestyle="None",
        markersize=22,
        alpha=0.5,
        markerfacecolor="red",
        label="Spikes (signal)",
    )
    leg = ax.legend(
        handles=[bulk_handle, spike_handle],
        loc="center right",
        frameon=True,
        fancybox=True,
        framealpha=0.8,
    )
    leg.get_frame().set_edgecolor("#b0b0b0")
    leg.get_frame().set_linewidth(2.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()

    # save requested output
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # also save a png automatically
    out_png = out.with_suffix(".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved {out}")
    print(f"Saved {out_png}")

    # =========================
    # Figure 2: top |eigenvalues| vs rank
    # =========================
    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    topk_abs = min(args.topk_abs, len(abs_full_eigs))
    ranks = np.arange(1, topk_abs + 1)

    ax.plot(ranks, abs_full_eigs[:topk_abs], color='k', marker='.', markersize=0.5, linewidth=2.0, label=r"$|\lambda_i|$ (full spectrum)")


    # if len(abs_ritz_eigs) > 0:
    #     k_ritz = min(len(abs_ritz_eigs), topk_abs)
    #     ax.plot(
    #         np.arange(1, k_ritz + 1),
    #         abs_ritz_eigs[:k_ritz],
    #         linestyle="None",
    #         marker="o",
    #         markersize=4,
    #         label="selected Ritz values",
    #     )

    # cutoff en d_1
    d1 = int(metrics["p"])
    cutoff = d1 + 0.5

    # fond rouge très léger à gauche
    ax.axvspan(0.5, cutoff, color="red", alpha=0.06, zorder=0)

    # barre verticale + légende
    ax.axvline(
        cutoff,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=r"cutoff in $d_1$",
    )

    ttl = args.title
    if ttl is None:
        ttl = (
            r"$|\lambda|$ vs rank, " 
            f"d={metrics['d']}, alpha={metrics['alpha']}, "
            f"$p_1$={metrics['rf_width']}"
        )
    ax.set_title(ttl, pad=18)

    ax.set_xlabel("Rank")
    ax.set_ylabel(r"$|\lambda|$")
    #ax.set_title((args.title + " — " if args.title else "") + r"Top $|\lambda|$ vs rank")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    out2_pdf = out.with_name(out.stem + "_topabs.pdf")
    out2_png = out.with_name(out.stem + "_topabs.png")

    fig.tight_layout()
    fig.savefig(out2_pdf)
    fig.savefig(out2_png, dpi=200)
    plt.close(fig)

    print(f"Saved {out2_pdf}")
    print(f"Saved {out2_png}")

    
    
    # # =========================
    # # Figure 3: sigma(Wx) vs projected latents
    # # =========================
    # fig3, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # idx_pre = np.arange(1, len(s_var) + 1)
    # idx_post = np.arange(1, len(post_proj_var) + 1)

    # ttl3 = args.title
    # if ttl3 is None:
    #     ttl3 = f"d={metrics['d']}, alpha={metrics['alpha']}, p1={metrics['rf_width']}"
    # fig3.suptitle(ttl3, y=1.02)

    # # variance
    # axes[0].plot(idx_pre, s_var, "o--", linewidth=1.5, markersize=3.5, label="pre")
    # axes[0].plot(idx_post, post_proj_var, "s-", linewidth=1.5, markersize=3.5, label="post")
    # axes[0].set_title("variance")
    # axes[0].set_xlabel("index")
    # axes[0].set_ylabel("var")
    # axes[0].legend(frameon=False)

    # # mean
    # axes[1].plot(idx_pre, s_mean, "o--", linewidth=1.5, markersize=3.5, label="pre")
    # axes[1].plot(idx_post, post_proj_mean, "s-", linewidth=1.5, markersize=3.5, label="post")
    # axes[1].set_title("mean")
    # axes[1].set_xlabel("index")
    # axes[1].set_ylabel("mean")
    # axes[1].legend(frameon=False)

    # out_stats = out.with_name(out.stem + "_projstats" + out.suffix)
    # out_stats_png = out_stats.with_suffix(".png")

    # fig3.tight_layout()
    # fig3.savefig(out_stats, dpi=200, bbox_inches="tight")
    # fig3.savefig(out_stats_png, dpi=200, bbox_inches="tight")
    # plt.close(fig3)

    # print(f"Saved {out_stats}")
    # print(f"Saved {out_stats_png}")

    # =========================
    # Figure 3: sigma(Wx) vs projected features
    # =========================
    fig3, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # pre = sigma(Wx), post = projected features Z
    pre_mean = obj["s_mean"].numpy()
    pre_var  = obj["s_var"].numpy()
    post_mean = obj["post_proj_mean"].numpy()
    post_var  = obj["post_proj_var"].numpy()

    x_pre = np.linspace(0.0, 1.0, len(pre_var))
    x_post = np.linspace(0.0, 1.0, len(post_var))

    ttl3 = args.title
    if ttl3 is None:
        ttl3 = f"d={metrics['d']}, alpha={metrics['alpha']}, p1={metrics['rf_width']}, d1={metrics['p']}"
    fig3.suptitle(ttl3, y=1.02)

    # variance
    axes[0].plot(x_pre, pre_var, "o", markersize=1.8, alpha=0.7, label="pre")
    axes[0].plot(x_post, post_var, "s-", linewidth=1.4, markersize=4.0, label="post")
    axes[0].set_title("variance")
    axes[0].set_xlabel("normalized index")
    axes[0].set_ylabel("var")
    axes[0].legend(frameon=False)

    # mean
    axes[1].plot(x_pre, pre_mean, "o", markersize=1.8, alpha=0.7, label="pre")
    axes[1].plot(x_post, post_mean, "s-", linewidth=1.4, markersize=4.0, label="post")
    axes[1].set_title("mean")
    axes[1].set_xlabel("normalized index")
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