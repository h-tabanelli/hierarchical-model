from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import NeurIPS plotting helpers
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# Prefer the repo utility path mentioned by the user/colleague.
try:
    from scripts.plotting.plot_utils import NeurIPSFigure, setup_style  # type: ignore
except Exception:
    # Fallback for local testing / uploaded helper file name.
    try:
        from _plot_utils import NeurIPSFigure, setup_style  # type: ignore
    except Exception:
        # Final fallback if the script is copied into the repo root or plots/.
        helper_candidates = [
            REPO_ROOT / "scripts" / "plotting",
            REPO_ROOT,
            Path(__file__).resolve().parent,
        ]
        for cand in helper_candidates:
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
        from _plot_utils import NeurIPSFigure, setup_style  # type: ignore


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RESULTS_ROOT = Path("results")
OUTDIR = Path("figures/paper")
OUTNAME = "fig2_gamma_scan_d400_eps05"

# Full-width, two-panel paper figure.
FIG_WIDTH = 1.0          # multiplier of NeurIPS text width
FIG_ASPECT = 0.42        # tuned for 2 horizontal panels
DRAFT = True            # False -> use LaTeX rendering from neurips.mplstyle
ALSO_PNG = True

# Canonical family for Fig. 2
EXP_BY_GAMMA = {
    0.0: "D400_eps05_g00_v2",
    0.2: "D400_eps05_g02_v2",
    0.4: "D400_eps05_g04_v2",
    0.6: "D400_eps05_g06_v2",
    0.8: "D400_eps05_g08_v2",
    1.0: "D400_eps05_g10_v2",
}

MODEL = "true"
G_NAME = "id"
B_MODE = "powerlaw_diag"
D = 400
EPS = 0.5

# Metrics to plot
METRIC_LEFT = "mse"     # could also be "nmse"
METRIC_RIGHT = "ovH"

# Error band = SEM across seeds
USE_SEM = True
BAND_ALPHA = 0.16
SEED_LINE_ALPHA = 0.14
LINEWIDTH = 2.0
MARKERSIZE = 4.5

# X limits; if None they are inferred from data.
X_LIM = (1.0, 3.5)

# Optional threshold marker for k=2, epsilon=0.5 -> alpha_th = 2.5
SHOW_THRESHOLD = True
THRESHOLD_ALPHA = 2.5
THRESHOLD_LABEL = r"$\alpha_{\mathrm{th}} = k+\varepsilon$"

# Legend
LEGEND_NCOL = 2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _maybe_filter(df: pd.DataFrame, col: str, value):
    if col not in df.columns:
        return df
    if isinstance(value, bool):
        return df[df[col].astype(bool) == bool(value)]
    return df[df[col] == value]



def _collect_metrics(exp_id: str) -> pd.DataFrame:
    base = RESULTS_ROOT / exp_id
    if not base.exists():
        raise FileNotFoundError(f"Missing results directory: {base}")

    rows: list[dict] = []
    for p in base.rglob("metrics.jsonl"):
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_path"] = str(p)
                rows.append(rec)

    if not rows:
        raise RuntimeError(f"No metrics found in {base}")

    df = pd.DataFrame(rows)
    if "alpha" in df.columns:
        df["alpha"] = df["alpha"].astype(float).round(10)
    return df



def _aggregate_by_alpha(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    def _sem(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return np.nan
        return float(np.std(x, ddof=1) / np.sqrt(x.size))

    g = (
        df.groupby("alpha")[value_col]
        .agg(["mean", "std", "count", _sem])
        .reset_index()
        .rename(columns={"mean": "y_mean", "std": "y_std", "count": "y_count", "_sem": "y_sem"})
        .sort_values("alpha")
    )
    return g



def _gamma_label(gamma: float) -> str:
    return rf"$\gamma={gamma:.1f}$"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Explicitly load the NeurIPS style and TeX rendering.
    setup_style(draft=DRAFT)
    plt.rcParams["text.usetex"] = not DRAFT

    # Collect all experiments first so we fail early if something is missing.
    data_by_gamma: dict[float, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for gamma, exp_id in EXP_BY_GAMMA.items():
        df = _collect_metrics(exp_id)
        df = _maybe_filter(df, "model", MODEL)
        df = _maybe_filter(df, "g_name", G_NAME)
        df = _maybe_filter(df, "B_mode", B_MODE)
        if "d" in df.columns:
            df = df[np.isclose(df["d"].astype(float), float(D))]
        if "eps" in df.columns:
            df = df[np.isclose(df["eps"].astype(float), float(EPS))]
        if "gamma" in df.columns:
            df = df[np.isclose(df["gamma"].astype(float), float(gamma))]

        if df.empty:
            raise RuntimeError(f"No rows left after filtering for exp_id={exp_id}")

        df = df.sort_values(["seed", "alpha"]).copy()
        g_left = _aggregate_by_alpha(df, METRIC_LEFT)
        g_right = _aggregate_by_alpha(df, METRIC_RIGHT)
        data_by_gamma[gamma] = (df, g_left, g_right)

    out_pdf = OUTDIR / f"{OUTNAME}.pdf"

    with NeurIPSFigure(
        width=FIG_WIDTH,
        aspect=FIG_ASPECT,
        ncols=2,
        draft=DRAFT,
        save=True,
        out_path=out_pdf,
        also_png=ALSO_PNG,
        sharex=True,
    ) as (fig, axes):
        ax1, ax2 = axes

        for gamma in sorted(data_by_gamma):
            df, g_left, g_right = data_by_gamma[gamma]
            label = _gamma_label(gamma)

            # Left panel: MSE
            for _, sub in df.groupby("seed"):
                ax1.plot(
                    sub["alpha"].to_numpy(float),
                    sub[METRIC_LEFT].to_numpy(float),
                    alpha=SEED_LINE_ALPHA,
                    linewidth=0.9,
                )

            line_left, = ax1.plot(
                g_left["alpha"].to_numpy(float),
                g_left["y_mean"].to_numpy(float),
                marker="o",
                ms=MARKERSIZE,
                linewidth=LINEWIDTH,
                label=label,
            )
            err = g_left["y_sem"].to_numpy(float) if USE_SEM else g_left["y_std"].to_numpy(float)
            lo = np.maximum(g_left["y_mean"].to_numpy(float) - err, 1e-12)
            hi = np.maximum(g_left["y_mean"].to_numpy(float) + err, lo * (1 + 1e-12))
            ax1.fill_between(
                g_left["alpha"].to_numpy(float),
                lo,
                hi,
                color=line_left.get_color(),
                alpha=BAND_ALPHA,
                linewidth=0,
            )

            # Right panel: overlap
            for _, sub in df.groupby("seed"):
                ax2.plot(
                    sub["alpha"].to_numpy(float),
                    sub[METRIC_RIGHT].to_numpy(float),
                    alpha=SEED_LINE_ALPHA,
                    linewidth=0.9,
                )

            line_right, = ax2.plot(
                g_right["alpha"].to_numpy(float),
                g_right["y_mean"].to_numpy(float),
                marker="o",
                ms=MARKERSIZE,
                linewidth=LINEWIDTH,
                label=label,
            )
            err = g_right["y_sem"].to_numpy(float) if USE_SEM else g_right["y_std"].to_numpy(float)
            lo = np.clip(g_right["y_mean"].to_numpy(float) - err, 0.0, 1.0)
            hi = np.clip(g_right["y_mean"].to_numpy(float) + err, 0.0, 1.0)
            ax2.fill_between(
                g_right["alpha"].to_numpy(float),
                lo,
                hi,
                color=line_right.get_color(),
                alpha=BAND_ALPHA,
                linewidth=0,
            )

        if SHOW_THRESHOLD:
            ax1.axvline(THRESHOLD_ALPHA, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
            ax2.axvline(THRESHOLD_ALPHA, color="black", linestyle="--", linewidth=1.1, alpha=0.8)

        ax1.set_xlabel(r"$\alpha = \log(n)/\log(d)$")
        ax2.set_xlabel(r"$\alpha = \log(n)/\log(d)$")
        ax1.set_ylabel(r"MSE")
        ax2.set_ylabel(r"Feature overlap on $h^{(1)}$")

        ax1.set_yscale("log")
        ax2.set_ylim(-0.02, 1.02)

        if X_LIM is not None:
            ax1.set_xlim(*X_LIM)
            ax2.set_xlim(*X_LIM)

        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)

        # One common legend on top.
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=LEGEND_NCOL,
            frameon=False,
            bbox_to_anchor=(0.5, 1.06),
        )

        if SHOW_THRESHOLD:
            ax1.text(
                THRESHOLD_ALPHA + 0.03,
                ax1.get_ylim()[1] / 1.6,
                THRESHOLD_LABEL,
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
            )

        fig.suptitle(rf"Fixed $d={D}$, $\varepsilon={EPS}$, varying $\gamma$", y=1.02)


if __name__ == "__main__":
    main()
