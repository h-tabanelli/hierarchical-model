from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
EXP_ID = "D400_eps05_g10"
SUMMARY_DIR = Path(f"summary/{EXP_ID}")   # <-- change to ..._partial or final
OUTDIR = Path(f"figures/{EXP_ID}")
OUTNAME = "fig_ab_nmse_and_overlapH1_vs_alpha_fan_by_seed"

MODEL = "true"     # "true" or "gauss"
METRIC = "mse"    # "nmse" or "mse"
OVERLAP_COL = "ovH"  # overlap on h^(1) = "ovH" in our pipeline

# If overlaps are close to 1, it's often better to plot 1-overlap on log-scale
PLOT_OVERLAP_AS_ERROR = False   # if True: plot (1-ovH); else plot ovH

# Fan plot style
SEED_ALPHA = 0.3     # transparency for per-seed lines
MEAN_LINEWIDTH = 2.2
BAND_ALPHA = 0.2
GRID_ALPHA = 0.01

X_LIM_ALPHA = None  # e.g. (1.0, 3.8)

d = 400
gamma = 1.0
eps = 0.5
k = 2

# =========================
# helpers
# =========================
def sem_from_std_count(std: np.ndarray, count: np.ndarray) -> np.ndarray:
    std = np.asarray(std, float)
    count = np.asarray(count, float)
    return std / np.sqrt(np.maximum(count, 1.0))

def first_decay(k=2, eps=0.5, gamma=0.0):
    d1 = int(np.round(d**eps))
    Cgamma = np.sum([i**(-2*gamma) for i in range(1, d1+1)])**(-1/2)
    return k - 2 * np.log(Cgamma) / np.log(d)

def last_decay(k=2, eps=0.5, gamma=0.0):
    d1 = int(np.round(d**eps))
    Cgamma = np.sum([i**(-2*gamma) for i in range(1, d1+1)])**(-1/2)
    return k + 2 * gamma * eps - 2 * np.log(Cgamma) / np.log(d)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    raw_path = SUMMARY_DIR / "raw_metrics.parquet"
    sum_path = SUMMARY_DIR / "summary.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path} (run aggregate_2layers.py first)")
    if not sum_path.exists():
        raise FileNotFoundError(f"Missing: {sum_path} (run aggregate_2layers.py first)")

    df = pd.read_parquet(raw_path)
    df = df[df["model"] == MODEL].copy()
    df = df.sort_values(["seed", "alpha"])

    # Build overlap y
    if PLOT_OVERLAP_AS_ERROR:
        df["ov_plot"] = 1.0 - df[OVERLAP_COL].astype(float)
        y_overlap_label = r"$1-\mathrm{overlap}(h^{(1)})$"
        overlap_log = True
    else:
        df["ov_plot"] = df[OVERLAP_COL].astype(float)
        y_overlap_label = r"$\mathrm{overlap}(h^{(1)})$"
        overlap_log = False

    # Summary (mean/std/count already computed)
    s = pd.read_csv(sum_path, dtype={"model": "string"})
    s = s[s["model"] == "true"]

    # Columns naming from aggregate_2layers.py
    y_mean = s[f"{METRIC}_mean"].to_numpy(float)
    y_std  = s.get(f"{METRIC}_std", pd.Series(np.nan, index=s.index)).to_numpy(float)
    y_cnt  = s.get(f"{METRIC}_count", pd.Series(1, index=s.index)).to_numpy(float)

    ov_mean_raw = s.get(f"{OVERLAP_COL}_mean", pd.Series(np.nan, index=s.index)).to_numpy(float)
    ov_std_raw  = s.get(f"{OVERLAP_COL}_std", pd.Series(np.nan, index=s.index)).to_numpy(float)
    ov_cnt      = s.get(f"{OVERLAP_COL}_count", pd.Series(1, index=s.index)).to_numpy(float)

    if PLOT_OVERLAP_AS_ERROR:
        ov_mean = 1.0 - ov_mean_raw
        ov_std  = ov_std_raw  # std doesn't change under affine shift
    else:
        ov_mean = ov_mean_raw
        ov_std  = ov_std_raw

    # SEM bands (if only 1 seed, std is NaN -> band ignored)
    y_sem = sem_from_std_count(y_std, y_cnt)
    ov_sem = sem_from_std_count(ov_std, ov_cnt)

    # x
    x = s["alpha"].to_numpy(float)

    # =========================
    # plotting
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)

    fig.suptitle(fr"$d={d},\ \epsilon={eps},\ \gamma={gamma}$", fontsize=14, y=1.08)

    # ---- Panel A: METRIC vs alpha (fan by seed) ----
    ax = axes[0]
    for seed, g in df.groupby("seed"):
        ax.plot(g["alpha"].to_numpy(float), g[METRIC].to_numpy(float), color='k', linewidth=0.5, alpha=SEED_ALPHA)

    ax.plot(x, y_mean, linewidth=MEAN_LINEWIDTH)
    if np.all(np.isfinite(y_sem)):
        ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=BAND_ALPHA)

    # ax.axvline(first_decay(k=2, eps=eps, gamma=gamma), color='r', linewidth=0.5, label='First Decay (Theoretical)')
    # ax.axvline(last_decay(k=2, eps=eps, gamma=gamma), color='g', linewidth=0.5, label='Last Decay (Theoretical)')
    # if gamma < 0.5:
    #     ax.plot(g["alpha"].to_numpy(float), 1-(Cgamma * g["alpha"].to_numpy(float)*(1-2*gamma)*np.log(d)/(2*gamma) ))
    # elif gamma >= 0.5:
    #     ax.plot(g["alpha"].to_numpy(float), g["alpha"].to_numpy(float)*(1-2*gamma)*np.log(d)/(2*gamma) )

    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(METRIC.upper())
    ax.set_yscale("log")
    if X_LIM_ALPHA is not None:
        ax.set_xlim(*X_LIM_ALPHA)
    ax.grid(True, alpha=GRID_ALPHA)
    # ax.legend()

    # ---- Panel B: overlap vs alpha (fan by seed) ----
    ax = axes[1]
    for seed, g in df.groupby("seed"):
        ax.plot(g["alpha"].to_numpy(float), g["ov_plot"].to_numpy(float), color='k', linewidth=0.5, alpha=SEED_ALPHA)

    ax.plot(x, ov_mean, linewidth=MEAN_LINEWIDTH)
    if np.all(np.isfinite(ov_sem)):
        ax.fill_between(x, ov_mean - ov_sem, ov_mean + ov_sem, alpha=BAND_ALPHA)

    # ax.axvline(first_decay(k=2, eps=eps, gamma=gamma), color='r', linewidth=0.5, label='First Decay (Theoretical)')
    # ax.axvline(last_decay(k=2, eps=eps, gamma=gamma), color='g', linewidth=0.5, label='Last Decay (Theoretical)')
    # ax.plot(g["alpha"].to_numpy(float), g["alpha"].to_numpy(float)*(1-2*gamma)*np.log(d)/(2*gamma))


    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(y_overlap_label)
    if overlap_log:
        ax.set_yscale("log")
    if X_LIM_ALPHA is not None:
        ax.set_xlim(*X_LIM_ALPHA)
    ax.grid(True, alpha=GRID_ALPHA)

    out_png = OUTDIR / f"{OUTNAME}_{MODEL}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{MODEL}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()