from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RESULTS_ROOT = Path("results")
EXP_ID = "D800_eps05_g10_id_true_cal"
OUTDIR = Path("figures") / EXP_ID
OUTNAME = "fig_rate_mse_ovH_d800_g10_direct"

# Filters
MODEL = "true"
G_NAME = "id"
B_MODE = "powerlaw_diag"
HEAD_MODE = "spectral_B"
LAYER1_MODE = "hermite_spectral"
CALIBRATE_OUTPUT = True   # set to None to ignore this filter
D = 800
EPS = 0.5
GAMMA = 1.0

# Plot choices
METRIC = "mse"           # e.g. mse, nmse, mse_scaled, nmse_scaled
OVERLAP_COL = "ovH"
X_LIM = (0.5, 3.1)
MSE_YLIM = None           # e.g. (1e-2, 2)
OV_YLIM = (0.0, 1.05)
TITLE = None
SHOW_SEED_CURVES = True

# Theoretical rate segment on left panel
DRAW_RATE = True
RATE_X0 = 2.55
RATE_X1 = 3.05
RATE_LABEL_X = 2.60
RATE_LABEL_DY = 1.12
RATE_TEXT = None          # if None, auto from gamma
RATE_LINEWIDTH = 3.0

# Visuals
FIGSIZE = (10.5, 4.2)
SEED_ALPHA = 0.25
BAND_ALPHA = 0.22
MARKER_SIZE = 4
DPI = 220


# =========================
# HELPERS
# =========================
def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).astype(bool)
    low = s.astype(str).str.lower().str.strip()
    return low.isin(["1", "true", "t", "yes", "y"])


def maybe_filter(df: pd.DataFrame, col: str, value):
    if value is None or col not in df.columns:
        return df
    if isinstance(value, bool):
        return df[_coerce_bool_series(df[col]) == value]
    return df[df[col] == value]


def load_metrics_from_results(results_root: Path, exp_id: str) -> pd.DataFrame:
    exp_dir = results_root / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {exp_dir}")

    rows: list[dict] = []
    metric_files = sorted(exp_dir.glob("chunk=*/seed=*/metrics.jsonl"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics.jsonl found under {exp_dir}")

    for mp in metric_files:
        with mp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    if not rows:
        raise RuntimeError(f"No readable metric rows found in {exp_dir}")

    df = pd.DataFrame(rows)
    return df


def build_summary(df: pd.DataFrame, metric: str, overlap_col: str) -> pd.DataFrame:
    agg = {
        metric: ["mean", "std", "count"],
        overlap_col: ["mean", "std", "count"],
    }
    s = df.groupby("alpha", as_index=False).agg(agg)
    s.columns = [
        "alpha" if c[0] == "alpha" else f"{c[0]}_{c[1]}"
        for c in s.columns.to_flat_index()
    ]
    s = s.sort_values("alpha").copy()
    return s


def sem(std: np.ndarray, count: np.ndarray) -> np.ndarray:
    count = np.maximum(count.astype(float), 1.0)
    std = np.nan_to_num(std.astype(float), nan=0.0)
    return std / np.sqrt(count)


def add_rate_segment(ax, s: pd.DataFrame, d: float, metric: str, gamma: float):
    if not DRAW_RATE:
        return
    if gamma <= 0.5:
        slope = -1.0 + 1.0 / (2.0 * gamma)
    else:
        slope = -1.0 + 1.0 / (2.0 * gamma)

    x0, x1 = RATE_X0, RATE_X1
    mid = 0.5 * (x0 + x1)
    y_mid = np.interp(mid, s["alpha"].to_numpy(), s[f"{metric}_mean"].to_numpy())
    if not np.isfinite(y_mid) or y_mid <= 0:
        return

    xs = np.array([x0, x1], dtype=float)
    ys = y_mid * (float(d) ** (slope * (xs - mid)))
    ax.plot(xs, ys, color="black", lw=RATE_LINEWIDTH, solid_capstyle="round")

    txt = RATE_TEXT if RATE_TEXT is not None else rf"slope $={slope:.2f}$"
    y_lab = y_mid * (float(d) ** (slope * (RATE_LABEL_X - mid))) * RATE_LABEL_DY
    ax.text(RATE_LABEL_X, y_lab, txt, color="black", fontsize=10)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_metrics_from_results(RESULTS_ROOT, EXP_ID)

    for col, value in [
        ("model", MODEL),
        ("g_name", G_NAME),
        ("B_mode", B_MODE),
        ("head_mode", HEAD_MODE),
        ("layer1_mode", LAYER1_MODE),
        ("calibrate_output", CALIBRATE_OUTPUT),
    ]:
        df = maybe_filter(df, col, value)

    for col, value in [("d", D), ("eps", EPS), ("gamma", GAMMA)]:
        if col in df.columns:
            df = df[np.isclose(df[col].astype(float), float(value))]

    if df.empty:
        raise RuntimeError("No rows left in direct metrics after filtering. Check config.")

    if METRIC not in df.columns:
        raise KeyError(f"Metric column '{METRIC}' not found. Available: {sorted(df.columns)}")
    if OVERLAP_COL not in df.columns:
        raise KeyError(f"Overlap column '{OVERLAP_COL}' not found. Available: {sorted(df.columns)}")

    df = df.sort_values(["seed", "alpha"]).copy()
    s = build_summary(df, METRIC, OVERLAP_COL)

    x = s["alpha"].to_numpy()
    y_m = s[f"{METRIC}_mean"].to_numpy()
    y_sem = sem(s[f"{METRIC}_std"].to_numpy(), s[f"{METRIC}_count"].to_numpy())
    ov_m = s[f"{OVERLAP_COL}_mean"].to_numpy()
    ov_sem = sem(s[f"{OVERLAP_COL}_std"].to_numpy(), s[f"{OVERLAP_COL}_count"].to_numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    if SHOW_SEED_CURVES and "seed" in df.columns:
        for seed, g in df.groupby("seed"):
            gg = g.sort_values("alpha")
            ax1.plot(gg["alpha"], gg[METRIC], lw=1.0, alpha=SEED_ALPHA)
            ax2.plot(gg["alpha"], gg[OVERLAP_COL], lw=1.0, alpha=SEED_ALPHA)

    ax1.plot(x, y_m, marker="o", ms=MARKER_SIZE, lw=2.0)
    lo = np.maximum(y_m - y_sem, 1e-16)
    hi = np.maximum(y_m + y_sem, lo * (1 + 1e-12))
    ax1.fill_between(x, lo, hi, alpha=BAND_ALPHA)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\alpha = \log(n)/\log(d)$")
    ax1.set_ylabel(METRIC.upper())
    ax1.set_xlim(*X_LIM)
    if MSE_YLIM is not None:
        ax1.set_ylim(*MSE_YLIM)
    add_rate_segment(ax1, s, float(D), METRIC, float(GAMMA))

    ax2.plot(x, ov_m, marker="o", ms=MARKER_SIZE, lw=2.0)
    ax2.fill_between(x, np.clip(ov_m - ov_sem, 0.0, 1.0), np.clip(ov_m + ov_sem, 0.0, 1.0), alpha=BAND_ALPHA)
    ax2.set_xlabel(r"$\alpha = \log(n)/\log(d)$")
    ax2.set_ylabel(OVERLAP_COL)
    ax2.set_xlim(*X_LIM)
    if OV_YLIM is not None:
        ax2.set_ylim(*OV_YLIM)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)

    if TITLE is None:
        title = rf"d={D}, eps={EPS}, gamma={GAMMA}, model={MODEL}, calib={CALIBRATE_OUTPUT}"
    else:
        title = TITLE
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()

    png = OUTDIR / f"{OUTNAME}.png"
    pdf = OUTDIR / f"{OUTNAME}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
