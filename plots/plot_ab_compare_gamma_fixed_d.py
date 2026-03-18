from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG (à éditer)
# =========================

d = 250
eps = 0.5
model = "true"       # "true" ou "gauss"
metric = "mse"       # "mse" ou "nmse"
overlap_col = "ovH"  # overlap sur h^(1)

# Liste (label, summary_dir) pour différents gamma, à d fixé
# -> adapte les exp_id / chemins à tes dossiers réels
EXP_SUMMARY_DIRS = [
    (r"$\gamma=0.4$", Path("summary/D120_eps05_g04")),
    (r"$\gamma=1.0$", Path("summary/D120_eps05_g10")),
    # (r"$\gamma=...$", Path("summary/...")),
]

# True: on force xlim à l'intersection des ranges alpha (utile si grilles différentes)
USE_COMMON_ALPHA_RANGE = True

OUTDIR = Path("figures/compare_gamma_fixed_d")
OUTNAME = "fig_ab_mse_overlapH1_vs_alpha_by_gamma_fixed_d"

LINEWIDTH = 2.2
GRID_ALPHA = 0.25


# =========================
# helpers
# =========================

@dataclass
class Curve:
    label: str
    alpha: np.ndarray
    y_mean: np.ndarray
    ov_mean: np.ndarray
    y_sem: np.ndarray
    ov_sem: np.ndarray
    alpha_min: float
    alpha_max: float


def load_mean_curve(summary_dir: Path, label: str) -> Curve:
    raw_path = summary_dir / "raw_metrics.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run aggregate_2layers.py first for {summary_dir}")

    df = pd.read_parquet(raw_path)
    df["model"] = df["model"].astype(str)
    df = df[df["model"] == model].copy()
    df = df.dropna(subset=["alpha", metric, overlap_col])

    # groupby alpha -> mean
    g = df.groupby("alpha").agg(
        y_mean=(metric, "mean"),
        y_std=(metric, "std"),
        y_count=(metric, "count"),
        ov_mean=(overlap_col, "mean"),
        ov_std=(overlap_col, "std"),
        ov_count=(overlap_col, "count"),
    ).reset_index().sort_values("alpha")

    # SEM (handle count=1 => std may be NaN)
    g["y_sem"] = g["y_std"] / np.sqrt(g["y_count"].clip(lower=1))
    g["ov_sem"] = g["ov_std"] / np.sqrt(g["ov_count"].clip(lower=1))

    alpha = g["alpha"].to_numpy(float)
    y_mean = g["y_mean"].to_numpy(float)
    ov_mean = g["ov_mean"].to_numpy(float)
    y_sem = g["y_sem"].to_numpy(float)
    ov_sem = g["ov_sem"].to_numpy(float)

    return Curve(
        label=label,
        alpha=alpha,
        y_mean=y_mean,
        ov_mean=ov_mean,
        y_sem=y_sem,
        ov_sem=ov_sem,
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    curves: list[Curve] = []
    for label, sdir in EXP_SUMMARY_DIRS:
        if not sdir.exists():
            print(f"[warn] skipping missing dir: {sdir}")
            continue
        curves.append(load_mean_curve(sdir, label))

    if not curves:
        raise SystemExit("No curves loaded (check EXP_SUMMARY_DIRS paths).")

    if USE_COMMON_ALPHA_RANGE:
        a0 = max(c.alpha_min for c in curves)
        a1 = min(c.alpha_max for c in curves)
        common_xlim = (a0, a1) if a0 < a1 else None
    else:
        common_xlim = None

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    fig.suptitle(fr"$d={d},\ \epsilon={eps}$", fontsize=14, y=1.08)

    ax0, ax1 = axes

    # Left: metric
    for c in curves:
        ax0.plot(c.alpha, c.y_mean, linewidth=LINEWIDTH, label=c.label)
        ax0.fill_between(c.alpha, c.y_mean - c.y_sem, c.y_mean + c.y_sem, alpha=0.18)

    ax0.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax0.set_ylabel(metric.upper())
    ax0.set_yscale("log")
    ax0.grid(True, alpha=GRID_ALPHA)

    # Right: overlap
    for c in curves:
        ax1.plot(c.alpha, c.ov_mean, linewidth=LINEWIDTH, label=c.label)
        ax1.fill_between(c.alpha, c.ov_mean - c.ov_sem, c.ov_mean + c.ov_sem, alpha=0.18)

    ax1.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax1.set_ylabel(r"overlap$(h^{(1)})$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(frameon=True, loc="lower right")

    if common_xlim is not None:
        ax0.set_xlim(*common_xlim)
        ax1.set_xlim(*common_xlim)

    out_png = OUTDIR / f"{OUTNAME}_{model}_d{d}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{model}_d{d}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()