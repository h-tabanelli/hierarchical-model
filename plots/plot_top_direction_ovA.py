# plots/posthoc_plot_topk_overlapA.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]  # parent de plots/ = racine du repo
sys.path.insert(0, str(REPO_ROOT))

import teacher
import measures as mps


# =========================
# CONFIG À ÉDITER
# =========================

MODEL = "true"          # "true" ou "gauss"
METRIC = "mse"          # "mse" ou "nmse"
K_TOP = 7             # top-K directions (modifiable rapidement)
d = 400
eps = 0.5

A_MODE_TEACHER = "sym_orth_frob"   # ton cas standard
USE_COMMON_ALPHA_RANGE = True      # intersection des ranges alpha entre expériences

RESULTS_ROOT = Path("results")
SUMMARY_ROOT = Path("summary")
OUTDIR = Path("figures/posthoc_topk_overlapA")
OUTNAME = "fig_ab_metric_and_ovA_topK_vs_alpha_by_exp"

# Mets ici tes expériences (exp_id = nom du dossier results/<exp_id> et summary/<exp_id>)
EXPS = [
    {"label": r"$\gamma=0.4$", "exp_id": "D400_eps05_g04_v2"},
    {"label": r"$\gamma=1.0$", "exp_id": "D400_eps05_g10_v2"},
    # {"label": r"...", "exp_id": "..."},
]



LINEWIDTH = 2.2
BAND_ALPHA = 0.18
GRID_ALPHA = 0.


# =========================
# Helpers
# =========================

@dataclass
class Curve:
    label: str
    alpha: np.ndarray
    y_mean: np.ndarray
    y_sem: np.ndarray
    ov_mean: np.ndarray
    ov_sem: np.ndarray
    a_min: float
    a_max: float


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _iter_estimate_files(exp_id: str):
    """Yield all estimates.pt for an experiment."""
    base = RESULTS_ROOT / exp_id
    yield from base.rglob("estimates.pt")


def _round_alpha(a: float) -> float:
    return float(np.round(a, 10))


@torch.no_grad()
def build_Atrue(d: int, p: int, seed: int) -> torch.Tensor:
    # Use CUDA if available to match how the teacher was generated during runs
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher["A"].detach().to("cpu")

@torch.no_grad()
def _build_Atrue_sym_orth_frob(d: int, p: int, seed: int) -> torch.Tensor:
    """
    Regenerate teacher A (sym_orth_frob) deterministically.
    IMPORTANT: use CUDA generator if available to match how A was generated in GPU runs.
    Returns A on CPU.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher["A"].detach().to("cpu")


@torch.no_grad()
def _ovA_topk(Ahat: torch.Tensor, Atrue: torch.Tensor, k: int) -> float:
    kk = int(min(k, Ahat.shape[0], Atrue.shape[0]))
    if kk <= 0:
        return float("nan")
    return float(bestK_overlap_frob(Ahat.to(dtype=torch.float32), Atrue.to(dtype=torch.float32), k=K_TOP))

def _flatten_A(A: torch.Tensor) -> torch.Tensor:
    # (p,d,d) -> (p, d*d)
    return A.reshape(A.shape[0], -1).to(torch.float32)

def _orthonormalize_rows(X: torch.Tensor) -> torch.Tensor:
    # QR on transpose gives orthonormal row basis
    Q, _ = torch.linalg.qr(X.T, mode="reduced")  # Q: (m, p)
    return Q.T.contiguous()  # (p, m)

def bestK_overlap_frob(Ahat: torch.Tensor, Atrue: torch.Tensor, k: int) -> float:
    """
    Best-K subspace overlap between spans(Atrue) and spans(Ahat), Frobenius inner product.
    Uses principal angles: overlap_K = mean_{i<=K} sigma_i^2 where sigma_i are svdvals(U V^T).
    Monotone in K and invariant to rotations/permutations.
    """
    Xh = _orthonormalize_rows(_flatten_A(Ahat))
    Xt = _orthonormalize_rows(_flatten_A(Atrue))
    S = Xt @ Xh.T  # (p,p)
    s = torch.linalg.svdvals(S)  # singular values in descending order
    s2 = (s ** 2).clamp(min=0.0, max=1.0)
    kk = int(min(k, s2.numel()))
    if kk <= 0:
        return float("nan")
    return float(s2[:kk].mean().cpu().item())

def make_curve(exp_id: str, label: str) -> Curve:
    # ---- Load raw metrics (MSE/NMSE etc.) ----
    raw_path = SUMMARY_ROOT / exp_id / "raw_metrics.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run: python3 cluster_tools/aggregate_2layers.py "
            f"--indir results/{exp_id} --outdir summary/{exp_id}"
        )

    dfm = pd.read_parquet(raw_path)
    dfm["model"] = dfm["model"].astype(str)
    dfm = dfm[dfm["model"] == MODEL].copy()
    dfm = dfm.dropna(subset=["alpha", METRIC, "seed"])
    dfm["alpha_key"] = dfm["alpha"].map(_round_alpha)

    # ---- Load artifacts and compute ovA_topK ----
    rows = []

    # cache Atrue per (d,p,seed)
    Atrue_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    n_files = 0
    for pth in _iter_estimate_files(exp_id):
        n_files += 1
        obj = torch.load(pth, map_location="cpu")

        # Basic fields
        model = str(obj.get("model", ""))
        if model != MODEL:
            continue

        alpha = obj.get("alpha", None)
        if alpha is None:
            continue
        alpha = _round_alpha(float(alpha))

        d = int(obj.get("d"))
        p = int(obj.get("p"))
        seed = int(obj.get("seed"))

        Ahat = obj["Ahat"].to("cpu")

        # teacher Atrue (cached)
        key = (d, p, seed)
        if key not in Atrue_cache:
            if A_MODE_TEACHER != "sym_orth_frob":
                raise ValueError("Ce script suppose A_MODE_TEACHER='sym_orth_frob'.")
            Atrue_cache[key] = _build_Atrue_sym_orth_frob(d=d, p=p, seed=seed)

        Atrue = Atrue_cache[key]
        ov = _ovA_topk(Ahat, Atrue, k=K_TOP)

        rows.append({
            "alpha_key": alpha,
            "seed": seed,
            "model": model,
            "ovA_topK": float(ov),
        })

    dfo = pd.DataFrame(rows)
    if dfo.empty:
        raise RuntimeError(
            f"No artifacts found for exp_id={exp_id}. "
            f"Did you generate tasks with --save_estimates and run them?"
        )

    # ---- Merge metrics + overlap by (alpha, seed, model) ----
    df = dfm.merge(dfo, on=["alpha_key", "seed", "model"], how="inner")
    if df.empty:
        raise RuntimeError(
            f"Merge empty for exp_id={exp_id}. "
            f"Check that artifacts and raw_metrics share same seeds/alphas and MODEL."
        )

    # ---- Aggregate mean + SEM over seeds (per alpha) ----
    out = []
    for a, g in df.groupby("alpha_key"):
        y = g[METRIC].to_numpy(float)
        ov = g["ovA_topK"].to_numpy(float)
        out.append({
            "alpha": float(a),
            "y_mean": float(np.nanmean(y)),
            "y_sem": _sem(y),
            "ov_mean": float(np.nanmean(ov)),
            "ov_sem": _sem(ov),
        })

    gg = pd.DataFrame(out).sort_values("alpha")
    alpha = gg["alpha"].to_numpy(float)

    return Curve(
        label=label,
        alpha=alpha,
        y_mean=gg["y_mean"].to_numpy(float),
        y_sem=gg["y_sem"].to_numpy(float),
        ov_mean=gg["ov_mean"].to_numpy(float),
        ov_sem=gg["ov_sem"].to_numpy(float),
        a_min=float(alpha.min()),
        a_max=float(alpha.max()),
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    curves = [make_curve(e["exp_id"], e["label"]) for e in EXPS]

    # common xlim if needed
    common_xlim = None
    if USE_COMMON_ALPHA_RANGE:
        a0 = max(c.a_min for c in curves)
        a1 = min(c.a_max for c in curves)
        if a0 < a1:
            common_xlim = (a0, a1)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    fig.suptitle(fr"{METRIC.upper()} & ovA(top-{K_TOP}) vs $\alpha$, $d=${d}, $\epsilon=${eps}", fontsize=14, y=1.08)

    ax0, ax1 = axes

    # Left: metric
    for c in curves:
        line, = ax0.plot(c.alpha, c.y_mean, linewidth=LINEWIDTH, label=c.label)
        col = line.get_color()
        if np.all(np.isfinite(c.y_sem)):
            ax0.fill_between(c.alpha, c.y_mean - c.y_sem, c.y_mean + c.y_sem,
                             alpha=BAND_ALPHA, color=col)

    ax0.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax0.set_ylabel(METRIC.upper())
    ax0.set_yscale("log")
    ax0.grid(True, alpha=GRID_ALPHA)

    # Right: overlap on A top-K
    for c in curves:
        line, = ax1.plot(c.alpha, c.ov_mean, linewidth=LINEWIDTH, label=c.label)
        col = line.get_color()
        if np.all(np.isfinite(c.ov_sem)):
            ax1.fill_between(c.alpha, c.ov_mean - c.ov_sem, c.ov_mean + c.ov_sem,
                             alpha=BAND_ALPHA, color=col)

    ax1.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax1.set_ylabel(fr"subspace overlap on $A$ (top-{K_TOP})")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(frameon=True, loc="lower right")

    if common_xlim is not None:
        ax0.set_xlim(*common_xlim)
        ax1.set_xlim(*common_xlim)

    out_png = OUTDIR / f"{OUTNAME}_{MODEL}_{METRIC}_ktop{K_TOP}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{MODEL}_{METRIC}_ktop{K_TOP}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()