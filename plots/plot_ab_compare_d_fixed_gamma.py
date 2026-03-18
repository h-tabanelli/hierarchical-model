from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG (à éditer à la main)
# =========================

k = 2
eps = 0.5
gamma = 1.0
model = "true"          # "true" ou "gauss"
metric = "mse"          # "mse" ou "nmse"
overlap_col = "ovH"     # overlap sur h^(1)

# Liste des expériences (summary dirs)
# -> adapte les exp_id / chemins à tes dossiers réels
EXP_SUMMARY_DIRS = [
    ("d=120", Path("summary/D120_eps05_g10")),     # gamma=1.0
    ("d=250", Path("summary/D250_eps05_g10")),     # gamma=1.0
    ("d=400", Path("summary/D400_eps05_g10")),     # gamma=1.0 (quand prêt)
]

# Si True: on force xlim à l'intersection des ranges alpha (utile car grilles différentes)
USE_COMMON_ALPHA_RANGE = True

OUTDIR = Path("figures/compare_d_fixed_gamma")
OUTNAME = "fig_ab_mse_overlapH1_vs_alpha_by_d_fixed_gamma"

# style
LINEWIDTH = 2.2
BAND_ALPHA = 0.18   # si tu actives des bands SEM (optionnel)
GRID_ALPHA = 0.25


# =========================
# helpers
# =========================

def first_decay(k=2, eps=0.5, gamma=0.0):
    if gamma <= 0.5:
        return k + (1-2*gamma)*eps
    elif gamma > 0.5:
        return k

def last_decay(k=2, eps=0.5, gamma=0.0):
    if gamma <= 0.5:
        return k + eps
    elif gamma > 0.5:
        return k + 2*gamma*eps

def first_decay_account_finite_d(d, k=2, eps=0.5, gamma=0.0):
    d1 = int(np.round(d**eps))
    k_eff = np.log(d*(d+1)/2)/np.log(d)
    Cgamma = np.sum([i**(-2*gamma) for i in range(1, d1+1)])**(-1/2)
    return k_eff - 2 * np.log(Cgamma) / np.log(d)


def last_decay_account_finite_d(d, k=2, eps=0.5, gamma=0.0):
    d1 = int(np.round(d**eps))
    k_eff = np.log(d*(d+1)/2)/np.log(d)
    Cgamma = np.sum([i**(-2*gamma) for i in range(1, d1+1)])**(-1/2)
    return k_eff + 2 * gamma * eps - 2 * np.log(Cgamma) / np.log(d)

def add_slope_guide_alpha(ax, *, d: int, s_n: float, x0: float, y0: float, dx: float,
                          color="k", linestyle="dotted", linewidth=1.2, alpha=0.9):
    """
    Draw a short guide segment corresponding to MSE ~ n^{s_n}.
    In alpha-coordinates: y(alpha+dx) = y(alpha) * d^{s_n * dx}.
    """
    x1 = x0 + dx
    y1 = y0 * (d ** (s_n * dx))
    ax.plot([x0, x1], [y0, y1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

@dataclass
class Curve:
    label: str
    d: int
    alpha: np.ndarray
    y_mean: np.ndarray
    ov_mean: np.ndarray
    y_sem: np.ndarray
    ov_sem: np.ndarray
    alpha_min: float
    alpha_max: float
    


def load_mean_curve(summary_dir: Path, label: str) -> Curve:
    """Charge raw_metrics.parquet et retourne les moyennes (par alpha) pour metric et overlap."""
    raw_path = summary_dir / "raw_metrics.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run aggregate_2layers.py first for {summary_dir}")

    df = pd.read_parquet(raw_path)

    # Filtre modèle (robuste : cast string)
    df["model"] = df["model"].astype(str)
    df = df[df["model"] == model].copy()
    df = df.dropna(subset=["alpha", metric, overlap_col])
    d_val = int(df["d"].iloc[0])

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
        d=d_val,
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

    # common alpha range if requested
    if USE_COMMON_ALPHA_RANGE:
        a0 = max(c.alpha_min for c in curves)
        a1 = min(c.alpha_max for c in curves)
        if a0 >= a1:
            print("[warn] no overlapping alpha range across d's; will not set common xlim.")
            common_xlim = None
        else:
            common_xlim = (a0, a1)
    else:
        common_xlim = None

    # ============ plot ============
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    fig.suptitle(fr"$\epsilon={eps},\ \gamma={gamma}$", fontsize=14, y=1.08)

    ax0, ax1 = axes

    # Left: metric
    for c in curves:
        d = c.d
        line, = ax0.plot(c.alpha, c.y_mean, linewidth=LINEWIDTH, label=c.label)
        col = line.get_color()
        ax0.fill_between(c.alpha, c.y_mean - c.y_sem, c.y_mean + c.y_sem, color=col, alpha=0.18)

        s_n = (1.0 / (2.0 * gamma)) - 1.0 #slope
        x0 = float(np.clip(2.0, c.alpha_min, c.alpha_max))
        y_curve = float(np.interp(x0, c.alpha, c.y_mean))
        y0 = 3.5 * y_curve / (d/100) # vertical offset so it "floats" (tune 1.2–3.0)
        add_slope_guide_alpha(ax0, d=d, s_n=s_n, x0=x0, y0=y0, dx=1.0, color=col,
                      linestyle="dotted", linewidth=1.2, alpha=0.9)
        # ax0.plot(c.alpha, d**((1/(2*gamma)-1)*c.alpha+1.5), linestyle='dashed', color='k', alpha=0.8, linewidth=0.8)
        # ax0.axvline(first_decay_account_finite_d(d=d, k=k, eps=eps, gamma=gamma), color='r', linestyle="dotted", linewidth=1, alpha=0.8)
        # ax0.axvline(last_decay_account_finite_d(d=d, k=k, eps=eps, gamma=gamma), color='g', linestyle="dotted", linewidth=1, alpha=0.8)


    ax0.axvline(first_decay(k=2, eps=eps, gamma=gamma), color='r', linewidth=0.5, label='First Decay (Theoretical)')
    ax0.axvline(last_decay(k=2, eps=eps, gamma=gamma), color='g', linewidth=0.5, label='Last Decay (Theoretical)')

    ax0.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax0.set_ylabel(metric.upper())
    ax0.set_yscale("log")
    ax0.set_ylim(0.05, 2)
    ax0.grid(True, alpha=GRID_ALPHA)

    # Right: overlap
    for c in curves:
        d = c.d
        line, = ax1.plot(c.alpha, c.ov_mean, linewidth=LINEWIDTH, label=c.label)
        col = line.get_color()
        ax1.fill_between(c.alpha, c.ov_mean - c.ov_sem, c.ov_mean + c.ov_sem, color=col, alpha=0.18)

        s_n = (1.0 / (2.0 * gamma)) #slope
        x0 = float(np.clip(2.0, c.alpha_min, c.alpha_max))
        y_curve = float(np.interp(x0, c.alpha, c.ov_mean))
        y0 = 3.5 * y_curve / (d/100) # vertical offset so it "floats" (tune 1.2–3.0)
        add_slope_guide_alpha(ax1, d=d, s_n=s_n, x0=x0, y0=y0, dx=1.0, color=col,
                      linestyle="dotted", linewidth=1.2, alpha=0.9)
        # ax1.axvline(first_decay_account_finite_d(d=d, k=k, eps=eps, gamma=gamma), color='r', linestyle="dotted", linewidth=1, alpha=0.8)
        # ax1.axvline(last_decay_account_finite_d(d=d, k=k, eps=eps, gamma=gamma), color='g', linestyle="dotted", linewidth=1, alpha=0.8)

    ax1.axvline(first_decay(k=2, eps=eps, gamma=gamma), color='r', linewidth=0.5, label='First Decay (Theoretical)')
    ax1.axvline(last_decay(k=2, eps=eps, gamma=gamma), color='g', linewidth=0.5, label='Last Decay (Theoretical)')

    ax1.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax1.set_ylabel(r"overlap$(h^{(1)})$")
    # ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylim(0.001, 1.02)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(frameon=True)

    if common_xlim is not None:
        ax0.set_xlim(*common_xlim)
        ax1.set_xlim(*common_xlim)

    out_png = OUTDIR / f"{OUTNAME}_{model}_g{gamma}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{model}_g{gamma}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()