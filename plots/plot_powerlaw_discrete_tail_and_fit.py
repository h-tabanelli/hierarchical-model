from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
EXP_ID = "D800_eps05_g10_id_true_cal"   # <-- change to your real exp_id
RESULTS_ROOT = Path("results")
OUTDIR = Path("figures") / EXP_ID
OUTNAME = "mse_vs_discrete_tail_and_fit"

MODEL = "true"
G_NAME = "id"
B_MODE = "powerlaw_diag"
HEAD_MODE = "spectral_B"
LAYER1_MODE = "hermite_spectral"
CALIBRATE_OUTPUT = True

D = 800
EPS = 0.5
GAMMA = 1.0
P = int(round(D ** EPS))

METRIC = "mse"          # e.g. "mse" or "mse_scaled"
OVERLAP_COL = "ovH"

# Display window
X_MIN = 0.5
X_MAX = 3.1

# Discrete theory: first recovered direction onset
ALPHA_FIRST = 2.0

# Compare to normalized discrete tail by default (no vertical free fit)
USE_NORMALIZED_THEORY = True

# Optional shape-only comparison with one multiplicative alignment
SHOW_ALIGNED_THEORY = False
ALIGN_X0 = 2.5
ALIGN_X1 = 3.0

# Linear fit window on log(MSE) / log(R_th)
FIT_X0 = 2.5
FIT_X1 = 3.00

# Plot style
FIGSIZE = (12.5, 4.8)
SEED_ALPHA = 0.22
BAND_ALPHA = 0.18
MARKER_SIZE = 5


# ============================================================
# IO helpers
# ============================================================
def maybe_filter(df: pd.DataFrame, col: str, value):
    if col not in df.columns:
        return df
    series = df[col]
    if isinstance(value, bool):
        if series.dtype == bool:
            return df[series == bool(value)]
        # robust fallback for ints/strings/bools mixed
        truthy = series.astype(str).str.lower().isin(["true", "1", "yes"])
        return df[truthy == bool(value)]
    return df[series == value]


def collect_metrics(results_root: Path, exp_id: str) -> pd.DataFrame:
    base = results_root / exp_id
    if not base.exists():
        raise FileNotFoundError(f"Missing results directory: {base}")

    rows = []
    for path in base.rglob("metrics.jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No metrics.jsonl found under {base}")
    return pd.DataFrame(rows)


# ============================================================
# Theory
# ============================================================
def m_theory(alpha: np.ndarray, p: int, gamma: float, d: float, alpha_first: float) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    expo = (alpha - alpha_first) / (2.0 * gamma)
    m = np.floor(d ** expo).astype(int)
    return np.clip(m, 0, p)


def r_theory(alpha: np.ndarray, p: int, gamma: float, d: float, alpha_first: float, normalize: bool) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    m = m_theory(alpha, p=p, gamma=gamma, d=d, alpha_first=alpha_first)
    weights = np.arange(1, p + 1, dtype=float) ** (-2.0 * gamma)
    cs = np.concatenate([[0.0], np.cumsum(weights)])
    total = cs[-1]
    out = np.empty_like(alpha, dtype=float)
    for i, mi in enumerate(m):
        out[i] = total - cs[int(mi)]
    if normalize and total > 0:
        out = out / total
    return out


def fit_vertical_scale(x: np.ndarray, y_ref: np.ndarray, y_model: np.ndarray, x0: float, x1: float) -> float:
    mask = (x >= x0) & (x <= x1) & (y_ref > 0) & (y_model > 0)
    if mask.sum() == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(y_ref[mask]) - np.log(y_model[mask]))))


# ============================================================
# Fit helpers
# ============================================================
def fit_logline(x: np.ndarray, y: np.ndarray, x0: float, x1: float, d: float):
    mask = (x >= x0) & (x <= x1) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None
    xs = x[mask]
    ly = np.log(y[mask])
    slope_alpha, intercept = np.polyfit(xs, ly, 1)  # ly = slope_alpha * x + intercept
    exponent_n = float(slope_alpha / np.log(d))
    return {
        "x": xs.tolist(),
        "intercept": float(intercept),
        "slope_alpha": float(slope_alpha),
        "exponent_n": float(exponent_n),
        "npts": int(mask.sum()),
    }


def eval_logline(x: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    return np.exp(fit["intercept"] + fit["slope_alpha"] * x)


# ============================================================
# Main
# ============================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = collect_metrics(RESULTS_ROOT, EXP_ID)

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
        raise RuntimeError("No rows left after filtering. Check config.")
    if METRIC not in df.columns:
        raise RuntimeError(f"Metric column '{METRIC}' not found.")
    if OVERLAP_COL not in df.columns:
        raise RuntimeError(f"Overlap column '{OVERLAP_COL}' not found.")

    df = df.sort_values(["seed", "alpha"]).copy()
    df = df[(df["alpha"] >= X_MIN) & (df["alpha"] <= X_MAX)].copy()
    if df.empty:
        raise RuntimeError("No rows left in display window.")

    s = df.groupby("alpha", as_index=False).agg({
        METRIC: ["mean", "std", "count"],
        OVERLAP_COL: ["mean", "std", "count"],
    })
    s.columns = [
        "alpha" if c[0] == "alpha" else f"{c[0]}_{c[1]}"
        for c in s.columns.to_flat_index()
    ]
    s = s.sort_values("alpha").copy()

    x = s["alpha"].to_numpy(dtype=float)

    y_m = s[f"{METRIC}_mean"].to_numpy(dtype=float)
    y_std = np.nan_to_num(s[f"{METRIC}_std"].to_numpy(dtype=float), nan=0.0)
    y_cnt = np.maximum(s[f"{METRIC}_count"].to_numpy(dtype=float), 1.0)
    y_sem = y_std / np.sqrt(y_cnt)

    ov_m = s[f"{OVERLAP_COL}_mean"].to_numpy(dtype=float)
    ov_std = np.nan_to_num(s[f"{OVERLAP_COL}_std"].to_numpy(dtype=float), nan=0.0)
    ov_cnt = np.maximum(s[f"{OVERLAP_COL}_count"].to_numpy(dtype=float), 1.0)
    ov_sem = ov_std / np.sqrt(ov_cnt)

    rth = r_theory(x, p=P, gamma=GAMMA, d=D, alpha_first=ALPHA_FIRST, normalize=USE_NORMALIZED_THEORY)
    mth = m_theory(x, p=P, gamma=GAMMA, d=D, alpha_first=ALPHA_FIRST)

    if SHOW_ALIGNED_THEORY:
        scale = fit_vertical_scale(x, y_m, rth, ALIGN_X0, ALIGN_X1)
        rth_aligned = scale * rth
    else:
        rth_aligned = None

    fit_mse = fit_logline(x, y_m, FIT_X0, FIT_X1, D)
    fit_rth = fit_logline(x, rth, FIT_X0, FIT_X1, D)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # Left panel: MSE + theory + fitted lines
    for seed, sub in df.groupby("seed"):
        ax1.plot(sub["alpha"].to_numpy(), sub[METRIC].to_numpy(), lw=1.1, alpha=SEED_ALPHA)

    line_mse, = ax1.plot(x, y_m, marker="o", ms=MARKER_SIZE, lw=2.2, label="MSE")
    lo = np.maximum(y_m - y_sem, 1e-16)
    hi = np.maximum(y_m + y_sem, lo * (1 + 1e-12))
    ax1.fill_between(x, lo, hi, color=line_mse.get_color(), alpha=BAND_ALPHA, linewidth=0)

    theory_label = r"$R_{\rm th}^{\rm norm}(\alpha)$" if USE_NORMALIZED_THEORY else r"$R_{\rm th}(\alpha)$"
    ax1.plot(x, rth, color="black", lw=2.8, label=theory_label)

    if rth_aligned is not None:
        ax1.plot(x, rth_aligned, color="black", lw=1.8, ls="--", alpha=0.9, label=r"aligned $R_{\rm th}(\alpha)$")

    mse_label = rf"MSE fit ($\beta={fit_mse['exponent_n']:.3f}$)"
    rth_label = rf"$R_{{\rm th}}$ fit ($\beta={fit_rth['exponent_n']:.3f}$)"

    x_fit_mse = np.linspace(FIT_X0, FIT_X1, 100)
    y_fit_mse = np.exp(fit_mse["intercept"] + fit_mse["slope_alpha"] * x_fit_mse)

    x_fit_rth = np.linspace(FIT_X0, FIT_X1, 100)
    y_fit_rth = np.exp(fit_rth["intercept"] + fit_rth["slope_alpha"] * x_fit_rth)
    
    if fit_mse is not None:
        ax1.plot(
            x_fit_mse,
            eval_logline(x_fit_mse, fit_mse),
            "--",
            lw=2.2,
            color=line_mse.get_color(),
            label=mse_label,
        )
    if fit_rth is not None:
        ax1.plot(
            x_fit_rth,
            eval_logline(x_fit_rth, fit_rth),
            "--",
            lw=2.2,
            color=line_mse.get_color(),
            label=rth_label,
        )

    ax1.axvspan(FIT_X0, FIT_X1, color="grey", alpha=0.08)
    ax1.set_yscale("log")
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax1.set_ylabel(METRIC)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)

    # Right panel: ovH + m_th
    for seed, sub in df.groupby("seed"):
        ax2.plot(sub["alpha"].to_numpy(), sub[OVERLAP_COL].to_numpy(), lw=1.1, alpha=SEED_ALPHA)

    line_ov, = ax2.plot(x, ov_m, marker="o", ms=MARKER_SIZE, lw=2.2, label=OVERLAP_COL)
    ax2.fill_between(
        x,
        np.clip(ov_m - ov_sem, 0.0, 1.0),
        np.clip(ov_m + ov_sem, 0.0, 1.0),
        color=line_ov.get_color(),
        alpha=BAND_ALPHA,
        linewidth=0,
    )

    ax2b = ax2.twinx()
    ax2b.step(x, mth, where="mid", color="black", lw=2.0, alpha=0.9, label=r"$m_{\rm th}(\alpha)$")
    ax2b.set_ylabel(r"$m_{\rm th}$")
    ax2b.set_ylim(0, P + 1)

    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax2.set_ylabel(OVERLAP_COL)
    ax2.grid(True, alpha=0.25)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right", fontsize=9)

    title = (
        f"{EXP_ID}\n"
        f"d={D}, eps={EPS}, gamma={GAMMA}, p={P}, alpha_first={ALPHA_FIRST:.2f}, "
        f"fit_window=[{FIT_X0:.2f}, {FIT_X1:.2f}]"
    )
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()

    png = OUTDIR / f"{OUTNAME}.png"
    pdf = OUTDIR / f"{OUTNAME}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Save CSV
    out_df = pd.DataFrame({
        "alpha": x,
        f"{METRIC}_mean": y_m,
        f"{METRIC}_sem": y_sem,
        f"{OVERLAP_COL}_mean": ov_m,
        f"{OVERLAP_COL}_sem": ov_sem,
        "m_th": mth,
        "r_th": rth,
    })
    out_df.to_csv(OUTDIR / f"{OUTNAME}.csv", index=False)

    # Save fit summary
    fit_summary = {
        "exp_id": EXP_ID,
        "metric": METRIC,
        "d": D,
        "eps": EPS,
        "gamma": GAMMA,
        "p": P,
        "alpha_first": ALPHA_FIRST,
        "fit_x0": FIT_X0,
        "fit_x1": FIT_X1,
        "use_normalized_theory": USE_NORMALIZED_THEORY,
        "mse_fit": fit_mse,
        "theory_fit": fit_rth,
    }
    with open(OUTDIR / f"{OUTNAME}_fit_summary.json", "w") as f:
        json.dump(fit_summary, f, indent=2)

    print(f"Saved:\n  {png}\n  {pdf}\n  {OUTDIR / (OUTNAME + '.csv')}\n  {OUTDIR / (OUTNAME + '_fit_summary.json')}")
    if fit_mse is not None:
        print(f"MSE fit over [{FIT_X0:.2f}, {FIT_X1:.2f}]: slope in alpha = {fit_mse['slope_alpha']:.6f}, exponent in n = {fit_mse['exponent_n']:.6f}")
    if fit_rth is not None:
        print(f"Theory fit over [{FIT_X0:.2f}, {FIT_X1:.2f}]: slope in alpha = {fit_rth['slope_alpha']:.6f}, exponent in n = {fit_rth['exponent_n']:.6f}")


if __name__ == "__main__":
    main()
