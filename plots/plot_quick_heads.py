from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

model = "true"
metric = "nmse"

# Mets ici les dossiers à fusionner pour chaque head mode
EXP_RESULT_GROUPS = [
    (
        "spectral_B",
        [
            Path("results/d200_eps04_g02_spectral_B"),
            Path("results/d200_eps04_g02_spectral_B_tail"),
            Path("results/d200_eps04_g02_spectral_B_tail2"),
        ],
    ),
    (
        "latent_rbf",
        [
            Path("results/d200_eps04_g02_latent_rbf"),
            Path("results/d200_eps04_g02_latent_rbf_tail"),
            Path("results/d200_eps04_g02_latent_rbf_tail2"),
        ],
    ),
    (
        "input_rbf",
        [
            Path("results/d200_eps04_g02_input_rbf"),
            Path("results/d200_eps04_g02_input_rbf_tail"),
            Path("results/d200_eps04_g02_input_rbf_tail2"),
        ],
    ),
]

USE_COMMON_ALPHA_RANGE = True

OUTDIR = Path("figures/d200_eps04_g02_kernel")
LINEWIDTH = 2.2
GRID_ALPHA = 0.0
BAND_ALPHA = 0.18

d = 200
eps = 0.4
gamma = 0.2


# =========================
# helpers
# =========================

@dataclass
class Curve:
    label: str
    alpha: np.ndarray
    y_mean: np.ndarray
    y_sem: np.ndarray
    alpha_min: float
    alpha_max: float


def load_raw_df_from_one_results_dir(exp_result_dir: Path) -> pd.DataFrame:
    rows = []
    for path in exp_result_dir.glob("chunk=*/seed=*/metrics.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No metrics found under {exp_result_dir}")
    return pd.DataFrame(rows)


def load_raw_df_from_many_results_dirs(exp_result_dirs: list[Path]) -> pd.DataFrame:
    dfs = []
    for rdir in exp_result_dirs:
        if not rdir.exists():
            print(f"[warn] skipping missing dir: {rdir}")
            continue
        try:
            dfs.append(load_raw_df_from_one_results_dir(rdir))
        except FileNotFoundError:
            print(f"[warn] no metrics found in: {rdir}")
            continue

    if not dfs:
        raise FileNotFoundError(f"No metrics found in any of: {exp_result_dirs}")

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # au cas où il y a des doublons exacts
    subset_cols = [c for c in ["alpha", "seed", "model", "head_mode", metric] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    return df


def load_mean_curve(exp_result_dirs: list[Path], label: str) -> Curve:
    df = load_raw_df_from_many_results_dirs(exp_result_dirs)

    df["model"] = df["model"].astype(str)
    df = df[df["model"] == model].copy()
    df = df.dropna(subset=["alpha", metric])

    g = df.groupby("alpha").agg(
        y_mean=(metric, "mean"),
        y_std=(metric, "std"),
        y_count=(metric, "count"),
    ).reset_index().sort_values("alpha")

    g["y_sem"] = g["y_std"] / np.sqrt(g["y_count"].clip(lower=1))
    g["y_sem"] = g["y_sem"].fillna(0.0)

    alpha = g["alpha"].to_numpy(float)
    y_mean = g["y_mean"].to_numpy(float)
    y_sem = g["y_sem"].to_numpy(float)

    return Curve(
        label=label,
        alpha=alpha,
        y_mean=y_mean,
        y_sem=y_sem,
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
    )


def make_single_plot(curve: Curve, common_xlim=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6), constrained_layout=True)
    fig.suptitle(fr"$d={d},\ \epsilon={eps},\ \gamma={gamma}$", fontsize=13)

    line, = ax.plot(curve.alpha, curve.y_mean, linewidth=LINEWIDTH, marker="o", label=curve.label)
    col = line.get_color()
    ax.fill_between(
        curve.alpha,
        curve.y_mean - curve.y_sem,
        curve.y_mean + curve.y_sem,
        color=col,
        alpha=BAND_ALPHA,
    )

    ax.axhline(1.0, color="k", linestyle="dashed", linewidth=0.9, alpha=0.8)

    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(metric.upper())
    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_title(curve.label)

    if common_xlim is not None:
        ax.set_xlim(*common_xlim)

    return fig, ax


def make_overlay_plot(curves: list[Curve], common_xlim=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.0), constrained_layout=True)
    fig.suptitle(fr"$d={d},\ \epsilon={eps},\ \gamma={gamma}$", fontsize=13)

    for c in curves:
        line, = ax.plot(c.alpha, c.y_mean, linewidth=LINEWIDTH, marker="o", label=c.label)
        col = line.get_color()
        ax.fill_between(
            c.alpha,
            c.y_mean - c.y_sem,
            c.y_mean + c.y_sem,
            color=col,
            alpha=0.12,
        )

    ax.axhline(1.0, color="k", linestyle="dashed", linewidth=0.9, alpha=0.8)
    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(metric.upper())
    ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(frameon=True)

    if common_xlim is not None:
        ax.set_xlim(*common_xlim)

    return fig, ax


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    curves: list[Curve] = []
    for label, rdirs in EXP_RESULT_GROUPS:
        try:
            curves.append(load_mean_curve(rdirs, label))
        except FileNotFoundError as e:
            print(f"[warn] skipping {label}: {e}")

    if not curves:
        raise SystemExit("No curves loaded. Check EXP_RESULT_GROUPS.")

    if USE_COMMON_ALPHA_RANGE:
        a0 = max(c.alpha_min for c in curves)
        a1 = min(c.alpha_max for c in curves)
        common_xlim = None if a0 >= a1 else (a0, a1)
    else:
        common_xlim = None

    for c in curves:
        fig, _ = make_single_plot(c, common_xlim=common_xlim)
        out_png = OUTDIR / f"{c.label}_{metric}_{model}.png"
        out_pdf = OUTDIR / f"{c.label}_{metric}_{model}.pdf"
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_png)
        print("Saved:", out_pdf)

    fig, _ = make_overlay_plot(curves, common_xlim=common_xlim)
    out_png = OUTDIR / f"overlay_{metric}_{model}.png"
    out_pdf = OUTDIR / f"overlay_{metric}_{model}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()
