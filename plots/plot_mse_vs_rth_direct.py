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

EXP_ID = "D800_eps05_g10_id_true_cal"   # <-- mets ici ton vrai exp_id
RESULTS_ROOT = Path("results")
OUTDIR = Path("figures") / EXP_ID
OUTNAME = "mse_vs_rth_direct"

MODEL = "true"
G_NAME = "id"
B_MODE = "powerlaw_diag"
HEAD_MODE = "spectral_B"
LAYER1_MODE = "hermite_spectral"
CALIBRATE_OUTPUT = True

D = 800
EPS = 0.5
GAMMA = 1.0

# métrique à tracer à gauche
METRIC = "mse"   # ou "mse_scaled" si tu veux tester, mais je déconseille pour l'instant

# on garde aussi le panel overlap global issu des metrics
OVERLAP_COL = "ovH"

# fenêtre en alpha
X_MIN = 0.5
X_MAX = 3.1

# nombre de directions du 2nd layer
P = int(round(D ** EPS))

# ordre "k" de la première layer (Hermite 2 ici)
K_LAYER = 2.0

# ----------------------------------------------------------------
# IMPORTANT :
# on absorbe les constantes / normalisations / C_gamma / finite-size
# dans un seul décalage ALPHA_FIRST :
#
# alpha où la PREMIÈRE direction commence à être récupérée.
#
# Tu peux le régler à la main. Commence par 2.0 ou 2.1 ou 2.2 selon le plot.
# ----------------------------------------------------------------
ALPHA_FIRST = 2.0

# fenêtre pour aligner verticalement R_th sur le MSE
# (on compare les formes, pas le préfacteur)
ALIGN_X0 = 2.2
ALIGN_X1 = 3.0

# style
FIGSIZE = (12, 4.6)
SEED_ALPHA = 0.22
BAND_ALPHA = 0.18
MARKER_SIZE = 5


# ============================================================
# IO
# ============================================================

def maybe_filter(df: pd.DataFrame, col: str, value):
    if col not in df.columns:
        return df
    if isinstance(value, bool):
        return df[df[col].astype(bool) == bool(value)]
    return df[df[col] == value]


def collect_metrics(results_root: Path, exp_id: str) -> pd.DataFrame:
    base = results_root / exp_id
    if not base.exists():
        raise FileNotFoundError(f"Missing results directory: {base}")

    rows = []
    for path in base.rglob("metrics.jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"No metrics.jsonl found under {base}")

    return pd.DataFrame(rows)


# ============================================================
# THEORY
# ============================================================

def m_theory(alpha: np.ndarray, p: int, gamma: float, d: float, alpha_first: float) -> np.ndarray:
    """
    Théorie discrète du nombre de directions récupérées :
        m_th(alpha) ≈ floor( d^((alpha - alpha_first)/(2 gamma)) )
    avec clipping dans [0, p].

    Toute la normalisation (y compris C_gamma, finite-size, etc.)
    est absorbée dans alpha_first.
    """
    alpha = np.asarray(alpha, dtype=float)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    expo = (alpha - alpha_first) / (2.0 * gamma)
    m = np.floor(d ** expo).astype(int)
    m = np.clip(m, 0, p)
    return m


def r_theory(alpha: np.ndarray, p: int, gamma: float, d: float, alpha_first: float) -> np.ndarray:
    """
    R_th(alpha) = sum_{j > m_th(alpha)} j^{-2 gamma}
    """
    alpha = np.asarray(alpha, dtype=float)
    m = m_theory(alpha, p=p, gamma=gamma, d=d, alpha_first=alpha_first)

    weights = np.arange(1, p + 1, dtype=float) ** (-2.0 * gamma)
    cs = np.concatenate([[0.0], np.cumsum(weights)])   # cs[k] = sum_{j<=k} w_j
    total = cs[-1]

    out = np.empty_like(alpha, dtype=float)
    for i, mi in enumerate(m):
        out[i] = total - cs[int(mi)]
    return out


def fit_vertical_scale(x: np.ndarray, y_ref: np.ndarray, y_model: np.ndarray,
                       x0: float, x1: float) -> float:
    """
    Trouve le meilleur facteur multiplicatif c > 0 tel que
        y_ref ≈ c * y_model
    en moyenne sur une fenêtre [x0, x1], en log-space.
    """
    mask = (x >= x0) & (x <= x1) & (y_ref > 0) & (y_model > 0)
    if mask.sum() == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(y_ref[mask]) - np.log(y_model[mask]))))


# ============================================================
# MAIN
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

    # agrégation simple
    s = df.groupby("alpha", as_index=False).agg({
        METRIC: ["mean", "std", "count"],
        OVERLAP_COL: ["mean", "std", "count"],
    })
    s.columns = [
        "alpha" if c[0] == "alpha" else f"{c[0]}_{c[1]}"
        for c in s.columns.to_flat_index()
    ]
    s = s.sort_values("alpha").copy()

    # fenêtre affichée
    s = s[(s["alpha"] >= X_MIN) & (s["alpha"] <= X_MAX)].copy()
    df = df[(df["alpha"] >= X_MIN) & (df["alpha"] <= X_MAX)].copy()

    if s.empty:
        raise RuntimeError("No alphas left in display window.")

    x = s["alpha"].to_numpy(dtype=float)
    y_m = s[f"{METRIC}_mean"].to_numpy(dtype=float)
    y_std = np.nan_to_num(s[f"{METRIC}_std"].to_numpy(dtype=float), nan=0.0)
    y_cnt = np.maximum(s[f"{METRIC}_count"].to_numpy(dtype=float), 1.0)
    y_sem = y_std / np.sqrt(y_cnt)

    ov_m = s[f"{OVERLAP_COL}_mean"].to_numpy(dtype=float)
    ov_std = np.nan_to_num(s[f"{OVERLAP_COL}_std"].to_numpy(dtype=float), nan=0.0)
    ov_cnt = np.maximum(s[f"{OVERLAP_COL}_count"].to_numpy(dtype=float), 1.0)
    ov_sem = ov_std / np.sqrt(ov_cnt)

    # théorie discrète
    rth = r_theory(x, p=P, gamma=GAMMA, d=D, alpha_first=ALPHA_FIRST)
    scale = fit_vertical_scale(x, y_m, rth, ALIGN_X0, ALIGN_X1)
    rth_scaled = scale * rth

    mth = m_theory(x, p=P, gamma=GAMMA, d=D, alpha_first=ALPHA_FIRST)

    # --------------------------------------------------------
    # plot
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # panel MSE
    for seed, sub in df.groupby("seed"):
        ax1.plot(
            sub["alpha"].to_numpy(),
            sub[METRIC].to_numpy(),
            lw=1.1,
            alpha=SEED_ALPHA,
        )

    line_mse, = ax1.plot(x, y_m, marker="o", ms=MARKER_SIZE, lw=2.2, label="MSE")
    lo = np.maximum(y_m - y_sem, 1e-16)
    hi = np.maximum(y_m + y_sem, lo * (1 + 1e-12))
    ax1.fill_between(x, lo, hi, color=line_mse.get_color(), alpha=BAND_ALPHA, linewidth=0)

    ax1.plot(x, rth_scaled, color="black", lw=3.0, label=r"$R_{\rm th}(\alpha)$ discret")

    ax1.set_yscale("log")
    ax1.set_xlim(X_MIN, X_MAX)
    ax1.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax1.set_ylabel(METRIC)
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.25)

    # panel overlap global + m_th
    for seed, sub in df.groupby("seed"):
        ax2.plot(
            sub["alpha"].to_numpy(),
            sub[OVERLAP_COL].to_numpy(),
            lw=1.1,
            alpha=SEED_ALPHA,
        )

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
    ax2b.step(x, mth, where="mid", color="black", lw=2.0, alpha=0.85, label=r"$m_{\rm th}(\alpha)$")
    ax2b.set_ylabel(r"$m_{\rm th}$")
    ax2b.set_ylim(0, P + 1)

    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax2.set_ylabel(OVERLAP_COL)
    ax2.grid(True, alpha=0.25)

    # légende combinée panel droit
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right")

    title = (
        f"{EXP_ID}\n"
        f"d={D}, eps={EPS}, gamma={GAMMA}, p={P}, "
        f"alpha_first={ALPHA_FIRST:.2f}"
    )
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()

    png = OUTDIR / f"{OUTNAME}.png"
    pdf = OUTDIR / f"{OUTNAME}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # export csv utile
    out_df = pd.DataFrame({
        "alpha": x,
        "mse_mean": y_m,
        "mse_sem": y_sem,
        "ovH_mean": ov_m,
        "ovH_sem": ov_sem,
        "m_th": mth,
        "r_th": rth,
        "r_th_scaled": rth_scaled,
    })
    out_df.to_csv(OUTDIR / f"{OUTNAME}.csv", index=False)

    print(f"Saved:\n  {png}\n  {pdf}\n  {OUTDIR / (OUTNAME + '.csv')}")


if __name__ == "__main__":
    main()