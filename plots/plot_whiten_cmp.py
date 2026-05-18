"""
Comparison plot: none / component / full whitening on second-stage RF.

3 original curves: relu_raw activation (empirical affine removal)
2 new curves:      relu_l1 activation (analytical L1 removal, simple head)
"""
import glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl_glob(path_glob: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(path_glob, recursive=True):
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise ValueError(f"No JSON rows found for glob: {path_glob}")
    return pd.DataFrame(rows)

GLOB = "results/whiten_cmp_d100/**/metrics.jsonl"
OUTDIR = Path("figures/whiten_cmp_d100")
MODEL = "true"
HEAD = "latent_rf_spectral"
LAYER1 = "rf_spectral"

PANELS = [
    ("nmse",         "NMSE"),
    ("nmse_scaled",  "NMSE scaled"),
    ("h2_pearson_r", "h² Pearson r"),
    ("h2_nmse_affine", "h² NMSE affine"),
]

# Each curve is (whiten_mode, rf2_activation, color, linestyle, label)
CURVES = [
    ("none",      "relu_raw", "tab:gray",   "-",  "none (relu_raw)"),
    ("component", "relu_raw", "tab:orange", "-",  "component-wise (relu_raw)"),
    ("full",      "relu_raw", "tab:blue",   "-",  "full ZCA (relu_raw)"),
    ("component", "relu_l1",  "tab:red",    "--", "component-wise (relu_l1)"),
    ("full",      "relu_l1",  "tab:cyan",   "--", "full ZCA (relu_l1)"),
]


def _plot_curves(ax, df, col):
    df = df.copy()
    df[col] = df[col].astype(float)
    for mode, act, color, ls, label in CURVES:
        mask = (df["rf2_whiten_mode"] == mode) & (df["rf2_activation"] == act)
        sub = df[mask].sort_values("alpha")
        if sub.empty:
            continue
        ax.plot(sub["alpha"], sub[col],
                marker="o", color=color, linestyle=ls, label=label, linewidth=2)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_jsonl_glob(GLOB)
    df = df[(df["model"] == MODEL) & (df["head_mode"] == HEAD) & (df["layer1_mode"] == LAYER1)].copy()
    df["alpha"] = df["alpha"].astype(float)

    # 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, ylabel) in zip(axes, PANELS):
        if col not in df.columns:
            ax.set_title(f"{ylabel}\n(not available)")
            continue
        _plot_curves(ax, df, col)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        r"Whitening mode comparison — $d=100$, $\epsilon=0.5$, $p_1=20000$, $p_2=512$",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"whiten_cmp_4panel.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Saved {p}")

    # 2-panel version (NMSE + h2_pearson_r)
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, ylabel) in zip(axes2, [("nmse", "NMSE"), ("h2_pearson_r", "h² Pearson r")]):
        if col not in df.columns:
            continue
        _plot_curves(ax, df, col)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig2.suptitle(
        r"Whitening mode comparison — $d=100$, $\epsilon=0.5$, $p_1=20000$, $p_2=512$",
        fontsize=12, y=1.02
    )
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"whiten_cmp_2panel.{ext}"
        fig2.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
