#!/usr/bin/env python3
"""Figure 3: Direction-wise recovery and theory comparison.

Panel A — Per-direction cos²(θᵢ) vs alpha, linear scale [0,1].
    Same quantity as Panel B but on linear scale to see the full [0,1] range.
    Direction i=1 (red) is easiest, i=p (blue) is hardest; sorted by the
    principal-angle SVD (rotation-invariant within the subspace).

Panel B — Per-direction 1 - cos(θᵢ) vs alpha.
    1 - sqrt(cos²(θᵢ)) on log scale.  A thick black rate segment shows the
    predicted 1/n decay: y_mid * d^(-(alpha - alpha_mid)).

Panel C — Normalized recovered rank vs theory staircase.
    rank_emp(α)/p vs m_th(α)/p, both in [0,1].

Usage
-----
  python plots/plot_fig3_directionwise_theory.py --draft
  python plots/plot_fig3_directionwise_theory.py
  python plots/plot_fig3_directionwise_theory.py --exp_id D400_eps05_g04_v2

WARNING: run on a CUDA machine.  A⋆ is regenerated from the training seed
using the GPU RNG; on CPU the generator state differs and the angles will be
wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import teacher  # noqa: E402
from _plot_utils import NeurIPSFigure  # noqa: E402

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_EXP_ID   = "D400_eps05_g04_v2"
DEFAULT_EXP_ID_C = "D800_eps05_g04_id_saveest"
MODEL          = "true"
A_MODE_TEACHER = "sym_orth_frob"

# directions to show in Panel B (1-indexed subset of 1..p)
DIRECTIONS_PANEL_B = [1, 3, 5, 8, 12, 15, 18, 20]

COS2_THRESHOLD = 0.5

# red→blue gradient (user palette)
_RED  = (255/255, 63/255,  69/255)
_BLUE = (95/255,  144/255, 255/255)
REDBLUE = mcolors.LinearSegmentedColormap.from_list("redblue", [_RED, _BLUE])

OUT_PATH = ROOT / "figures" / "fig3_directionwise_theory.pdf"


# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------

def alpha_th(i: int, gamma: float, eps: float, d: float) -> float:
    if gamma == 0.0:
        return 2.0 + eps
    elif gamma < 0.5:
        return 2.0 + (1.0 - 2.0 * gamma) * eps + 2.0 * gamma * np.log(i) / np.log(d)
    else:
        return 2.0 + 2.0 * gamma * np.log(i) / np.log(d)


def m_theory(alpha: np.ndarray, p: int, gamma: float, d: float,
             alpha_first: float) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    expo  = (alpha - alpha_first) / (2.0 * gamma)
    m     = np.floor(d ** expo).astype(int)
    return np.clip(m, 0, p)


# ---------------------------------------------------------------------------
# A* reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_Atrue_sym_orth_frob(d: int, p: int, seed: int) -> torch.Tensor:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher["A"].detach().to("cpu")


# ---------------------------------------------------------------------------
# cos² of principal angles  (Panels A and B)
# ---------------------------------------------------------------------------

def _flatten_A(A: torch.Tensor) -> torch.Tensor:
    return A.reshape(A.shape[0], -1).to(torch.float32)


def _orthonormalize_rows(X: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(X.T, mode="reduced")
    return Q.T.contiguous()


@torch.no_grad()
def principal_cos2(Ahat: torch.Tensor, Atrue: torch.Tensor) -> np.ndarray:
    Xh = _orthonormalize_rows(_flatten_A(Ahat))
    Xt = _orthonormalize_rows(_flatten_A(Atrue))
    S  = Xt @ Xh.T
    s  = torch.linalg.svdvals(S)
    return (s ** 2).clamp(0.0, 1.0).cpu().numpy()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_cos2_table(exp_dir: Path) -> pd.DataFrame:
    """Columns: alpha, seed, i (1-indexed, sorted by principal angle ↓), cos2."""
    Atrue_cache: dict[tuple, torch.Tensor] = {}
    rows: list[dict] = []

    for pth in sorted(exp_dir.rglob("estimates.pt")):
        try:
            obj = torch.load(pth, map_location="cpu")
        except Exception:
            continue
        if str(obj.get("model", "")) != MODEL:
            continue
        alpha = obj.get("alpha")
        if alpha is None:
            continue

        d    = int(obj["d"])
        p    = int(obj["p"])
        seed = int(obj["seed"])
        Ahat = obj["Ahat"].to("cpu")

        key = (d, p, seed)
        if key not in Atrue_cache:
            Atrue_cache[key] = build_Atrue_sym_orth_frob(d=d, p=p, seed=seed)

        s2 = principal_cos2(Ahat, Atrue_cache[key])
        a_r = round(float(alpha), 8)
        for i, v in enumerate(s2):
            rows.append({"alpha": a_r, "seed": seed, "i": i + 1, "cos2": float(v)})

    if not rows:
        raise RuntimeError(f"No usable estimates.pt under {exp_dir}.")
    return pd.DataFrame(rows)


def _agg_mean_sem(df: pd.DataFrame, val_col: str, group_col: str) -> pd.DataFrame:
    def _sem(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan

    return (
        df.groupby(["alpha", group_col])[val_col]
          .agg(["mean", _sem])
          .reset_index()
          .rename(columns={"mean": f"{val_col}_mean", "_sem": f"{val_col}_sem"})
          .sort_values([group_col, "alpha"])
    )


def compute_recovered_rank(cos2_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (alpha, seed), sub in cos2_df.groupby(["alpha", "seed"]):
        rows.append({"alpha": alpha, "seed": seed,
                     "rank_emp": int((sub["cos2"] > threshold).sum())})
    return pd.DataFrame(rows)


def collect_ovH_table(exp_dir: Path) -> pd.DataFrame:
    """Load ovH from metrics.jsonl when estimates.pt are not available.

    Returns a DataFrame with the same interface as the output of
    compute_recovered_rank but using ovH (already in [0,1]) in place of
    rank_emp/p, so plot_panel_c can consume it directly as a fraction.
    """
    import json
    rows: list[dict] = []
    for mf in sorted(exp_dir.rglob("metrics.jsonl")):
        for line in open(mf):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("model") == MODEL and "ovH" in row:
                    rows.append({
                        "alpha": round(float(row["alpha"]), 8),
                        "seed":  int(row["seed"]),
                        "ovH":   float(row["ovH"]),
                    })
            except Exception:
                pass
    if not rows:
        raise RuntimeError(f"No usable metrics.jsonl under {exp_dir}.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel_a(ax: plt.Axes, agg: pd.DataFrame,
                 directions: list[int], p: int) -> None:
    """Panel A: per-direction cos²(θᵢ) vs alpha, linear scale [0,1]."""
    usetex = plt.rcParams.get("text.usetex", False)
    n_dirs = len(directions)
    colors = [REDBLUE(v) for v in np.linspace(0.0, 1.0, n_dirs)]

    for i, col in zip(directions, colors):
        sub = agg[agg["i"] == i].sort_values("alpha")
        if sub.empty:
            continue
        x   = sub["alpha"].to_numpy(float)
        y   = sub["cos2_mean"].to_numpy(float)
        e   = sub["cos2_sem"].to_numpy(float)
        ax.plot(x, y, color=col, linewidth=1.2)
        ax.fill_between(x, np.clip(y - e, 0, 1), np.clip(y + e, 0, 1),
                        color=col, alpha=0.12, linewidth=0)

    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\alpha$")
    ylabel = r"${\rm cos}^2(\theta_i)$" if usetex else r"${\rm cos}^2(\theta_i)$"
    ax.set_ylabel(ylabel)


def plot_panel_b(ax: plt.Axes, agg: pd.DataFrame,
                 directions: list[int], d: float) -> None:
    """Panel B: per-direction 1-cos(θᵢ) on log scale + rate segment."""
    usetex  = plt.rcParams.get("text.usetex", False)
    n_dirs  = len(directions)
    colors  = [REDBLUE(v) for v in np.linspace(0.0, 1.0, n_dirs)]

    # collect all valid (alpha, y) for rate-segment anchor
    all_x, all_y = [], []

    for idx, (i, col) in enumerate(zip(directions, colors)):
        sub = agg[agg["i"] == i].sort_values("alpha")
        if sub.empty:
            continue
        x   = sub["alpha"].to_numpy(float)
        c2  = sub["cos2_mean"].to_numpy(float)
        e2  = sub["cos2_sem"].to_numpy(float)

        y   = 1.0 - np.sqrt(np.clip(c2, 0, 1))
        lbl = rf"$i={i}$" if usetex else f"i={i}"
        ax.plot(x, y, color=col, linewidth=1.2, label=lbl)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    # rate segment: y = y_mid * d^(-(alpha - alpha_mid))
    if all_x and all_y:
        ax_finite = [(xx, yy) for xx, yy in zip(all_x, all_y)
                     if np.isfinite(yy) and yy > 0]
        if ax_finite:
            xs_all, ys_all = zip(*ax_finite)
            alpha_mid = float(np.median(xs_all))
            # anchor at median-alpha, geometric mean of y-values at that alpha
            near = [(xx, yy) for xx, yy in zip(xs_all, ys_all)
                    if abs(xx - alpha_mid) < 0.15]
            if near:
                _, ys_near = zip(*near)
                y_mid = float(np.exp(np.mean(np.log(np.clip(ys_near, 1e-12, None)))))
            else:
                y_mid = float(np.median(ys_all))

            span  = 0.6
            xs_seg = np.linspace(alpha_mid - span / 2, alpha_mid + span / 2, 80) + 0.2
            ys_seg = y_mid * (d ** (-(xs_seg - alpha_mid))) * 0.1
            ax.plot(xs_seg, ys_seg, color="black", linewidth=1.5, zorder=5)
            lbl_x = xs_seg[-1] - 0.88
            lbl_y = ys_seg[-1] * 8.2
            rate_lbl = r"$~ 1/n$" if usetex else r"$\sim 1/n$"
            ax.text(lbl_x, lbl_y, rate_lbl, va="center", fontsize=7)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ylabel = r"$1 - {\rm cos}(\theta_i)$" if usetex else r"$1-{\rm cos}(\theta_i)$"
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=6, frameon=True,
              loc="lower left", labelspacing=0.2)


def plot_panel_c(ax: plt.Axes, rank_df: pd.DataFrame,
                 p: int, d: float, gamma: float, alpha_first: float) -> None:
    """Panel C: normalized recovered rank in [0,1] vs m_th/p staircase.

    rank_df may contain either a 'rank_emp' column (from estimates.pt) or an
    'ovH' column (from metrics.jsonl).  Both are normalized to [0,1].
    """
    usetex = plt.rcParams.get("text.usetex", False)

    if "ovH" in rank_df.columns:
        val_col = "ovH"
        normalizer = 1.0
    else:
        val_col = "rank_emp"
        normalizer = float(p)

    s = (
        rank_df.groupby("alpha")[val_col]
               .agg(["mean", "std", "count"])
               .reset_index()
               .sort_values("alpha")
    )
    x   = s["alpha"].to_numpy(float)
    y   = s["mean"].to_numpy(float) / normalizer
    sem = s["std"].to_numpy(float) / np.sqrt(np.maximum(s["count"].to_numpy(float), 1)) / normalizer

    emp_col = _BLUE
    ax.plot(x, y, color=emp_col, linewidth=1.6,
            label="Empirical")
    ax.fill_between(x, np.clip(y - sem, 0, 1), np.clip(y + sem, 0, 1),
                    color=emp_col, alpha=0.2, linewidth=0)

    print("max_alpha =", x.max())

    x_th = np.linspace(float(x.min()), float(x.max()), 500)
    mth  = m_theory(x_th, p=p, gamma=gamma, d=d, alpha_first=alpha_first) / p
    th_lbl = r"$m_{\rm th}(\alpha)$" if usetex else r"$m_{\rm th}(\alpha)$"
    ax.step(x_th, mth, where="post", color="black",
            linewidth=1.8, label=th_lbl, alpha=0.85)

    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(r"$\alpha$")
    ylabel = r"Overlap $h^{(1)}$" if usetex else r"Overlap $h^{(1)}$"
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, loc="lower right")
    ax.set_xlim(2, x.max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_id",   default=DEFAULT_EXP_ID)
    ap.add_argument("--exp_id_c", default=DEFAULT_EXP_ID_C,
                    help="Experiment for panel C (defaults to D800). "
                         "Uses ovH from metrics.jsonl when estimates.pt are absent.")
    ap.add_argument("--draft",  action="store_true")
    ap.add_argument("--out",    default=None)
    args = ap.parse_args()

    exp_dir = ROOT / "results" / args.exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    exp_dir_c = ROOT / "results" / args.exp_id_c
    if not exp_dir_c.exists():
        raise FileNotFoundError(f"Panel-C experiment directory not found: {exp_dir_c}")

    out = Path(args.out) if args.out else OUT_PATH

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — Atrue regenerated on CPU, angles may be wrong.")

    # infer parameters
    sample_pt = next(exp_dir.rglob("estimates.pt"), None)
    if sample_pt is None:
        raise RuntimeError(f"No estimates.pt found under {exp_dir}")
    sample = torch.load(sample_pt, map_location="cpu")
    d   = int(sample["d"])
    p   = int(sample["p"])
    eps = round(np.log(p) / np.log(d), 6)

    import glob, json
    gamma = None
    for mf in exp_dir.rglob("metrics.jsonl"):
        for line in open(mf):
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    if "gamma" in row:
                        gamma = float(row["gamma"])
                        break
                except Exception:
                    pass
        if gamma is not None:
            break
    if gamma is None:
        raise RuntimeError("Could not infer gamma from metrics.jsonl.")

    a_first = alpha_th(i=1, gamma=gamma, eps=eps, d=float(d))
    print(f"d={d}, p={p}, eps={eps:.3f}, gamma={gamma}, alpha_first={a_first:.3f}")

    print("Loading estimates.pt files for panels A/B (may take a minute)...")
    cos2_df  = collect_cos2_table(exp_dir)
    agg_cos2 = _agg_mean_sem(cos2_df, "cos2", "i")

    # Panel C: use exp_id_c (d=800 by default), loading ovH from metrics.jsonl
    print(f"Loading panel-C data from {args.exp_id_c}...")
    sample_c = next(exp_dir_c.rglob("metrics.jsonl"), None)
    if sample_c is None:
        raise RuntimeError(f"No metrics.jsonl found under {exp_dir_c}")
    row_c = None
    for line in open(sample_c):
        line = line.strip()
        if line:
            try: row_c = json.loads(line); break
            except Exception: pass
    d_c   = int(row_c["d"])
    p_c   = int(row_c["p"])
    eps_c = round(np.log(p_c) / np.log(d_c), 6)
    gamma_c = float(row_c["gamma"])
    a_first_c = alpha_th(i=1, gamma=gamma_c, eps=eps_c, d=float(d_c))
    print(f"Panel-C: d={d_c}, p={p_c}, eps={eps_c:.3f}, gamma={gamma_c}, "
          f"alpha_first={a_first_c:.3f}")

    rank_df_c = collect_ovH_table(exp_dir_c)

    dirs_b = [i for i in DIRECTIONS_PANEL_B if i <= p]

    with NeurIPSFigure(
        width=1.0, ncols=3, nrows=1, aspect=0.8,
        draft=args.draft, save=True, out_path=out,
    ) as (fig, axes):
        plot_panel_a(axes[0], agg_cos2, dirs_b, p)
        plot_panel_b(axes[1], agg_cos2, dirs_b, float(d))
        plot_panel_c(axes[2], rank_df_c, p_c, float(d_c), gamma_c, a_first_c)


if __name__ == "__main__":
    main()
