from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- make repo root importable (teacher/measures live at repo root) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import teacher  # noqa: E402


# =========================
# CONFIG
# =========================

MODEL = "true"
P_MAX = 20                 # number of curves you want
A_MODE_TEACHER = "sym_orth_frob"

RESULTS_ROOT = Path("results")
OUTDIR = Path("figures/principal_angles")
OUTNAME = "ovA_principal_angles_cos2_vs_alpha"

# experiments to compare (one plot per exp)
EXPS = [
    {"label": r"$\gamma=0.4$", "exp_id": "D400_eps05_g04_v2"},
    {"label": r"$\gamma=1.0$", "exp_id": "D400_eps05_g10_v2"},
]

# plotting style
LINEWIDTH = 1.8
BAND_ALPHA = 0.15
GRID_ALPHA = 0.


# =========================
# helpers
# =========================

def _round_alpha(a: float) -> float:
    return float(np.round(a, 10))

def _iter_estimates(exp_id: str):
    base = RESULTS_ROOT / exp_id
    yield from base.rglob("estimates.pt")

def _flatten_A(A: torch.Tensor) -> torch.Tensor:
    # A: (p,d,d) -> (p, d*d)
    return A.reshape(A.shape[0], -1).to(torch.float32)

def _orthonormalize_rows(X: torch.Tensor) -> torch.Tensor:
    # returns row-orthonormal basis (p, m)
    Q, _ = torch.linalg.qr(X.T, mode="reduced")  # Q: (m, p)
    return Q.T.contiguous()

@torch.no_grad()
def principal_cos2(Ahat: torch.Tensor, Atrue: torch.Tensor) -> np.ndarray:
    """
    Returns cos^2 of principal angles between spans(Atrue) and spans(Ahat),
    in Frobenius space, sorted decreasing.
    """
    Xh = _orthonormalize_rows(_flatten_A(Ahat))
    Xt = _orthonormalize_rows(_flatten_A(Atrue))
    S = Xt @ Xh.T                 # (p,p)
    s = torch.linalg.svdvals(S)   # singular values sorted decreasing
    s2 = (s**2).clamp(0.0, 1.0).cpu().numpy()
    return s2

@torch.no_grad()
def build_Atrue_sym_orth_frob(d: int, p: int, seed: int) -> torch.Tensor:
    """
    IMPORTANT: use CUDA generator if available to match GPU-generated teachers.
    Returns Atrue on CPU.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher["A"].detach().to("cpu")


def collect_cos2_table(exp_id: str) -> pd.DataFrame:
    """
    Returns a tidy dataframe with columns:
      alpha, seed, i (1..p), cos2
    """
    # cache Atrue per (d,p,seed)
    Atrue_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    rows = []
    for pth in _iter_estimates(exp_id):
        obj = torch.load(pth, map_location="cpu")

        if str(obj.get("model", "")) != MODEL:
            continue

        alpha = obj.get("alpha", None)
        if alpha is None:
            continue
        alpha = _round_alpha(float(alpha))

        d = int(obj["d"])
        p = int(obj["p"])
        seed = int(obj["seed"])

        Ahat = obj["Ahat"].to("cpu")

        key = (d, p, seed)
        if key not in Atrue_cache:
            if A_MODE_TEACHER != "sym_orth_frob":
                raise ValueError("Script assumes A_MODE_TEACHER='sym_orth_frob'.")
            Atrue_cache[key] = build_Atrue_sym_orth_frob(d=d, p=p, seed=seed)

        Atrue = Atrue_cache[key]
        s2 = principal_cos2(Ahat, Atrue)  # length p

        # keep only first P_MAX curves
        m = min(int(P_MAX), int(s2.shape[0]))
        for i in range(m):
            rows.append({"alpha": alpha, "seed": seed, "i": i + 1, "cos2": float(s2[i])})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No estimates.pt found or no rows collected for exp_id={exp_id}. "
            f"Check RESULTS_ROOT/results/{exp_id} and that artifacts exist."
        )
    return df


def mean_sem_by_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: tidy alpha/seed/i/cos2
    Output: alpha/i -> mean, sem
    """
    def _sem(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return np.nan
        return float(np.std(x, ddof=1) / np.sqrt(x.size))

    g = (
        df.groupby(["alpha", "i"])["cos2"]
          .agg(["mean", _sem, "count"])
          .reset_index()
          .rename(columns={"mean": "cos2_mean", "_sem": "cos2_sem"})
          .sort_values(["i", "alpha"])
    )
    return g


def plot_one_exp(exp_id: str, label: str):
    df = collect_cos2_table(exp_id)
    g = mean_sem_by_alpha(df)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.2), constrained_layout=True)
    ax.set_title(f"{label}  (exp_id={exp_id})", fontsize=13)

    # One curve per i
    for i in sorted(g["i"].unique()):
        gi = g[g["i"] == i]
        x = gi["alpha"].to_numpy(float)
        y = gi["cos2_mean"].to_numpy(float)
        e = gi["cos2_sem"].to_numpy(float)

        line, = ax.plot(x, y, linewidth=LINEWIDTH, label=f"i={int(i)}")
        col = line.get_color()
        if np.all(np.isfinite(e)):
            ax.fill_between(x, y - e, y + e, alpha=BAND_ALPHA, color=col)

    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(r"$\cos^2(\theta_i)$  (principal angles)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=GRID_ALPHA)

    # legend outside
    ax.legend(ncol=2, fontsize=9, frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))

    out_png = OUTDIR / f"{OUTNAME}_{exp_id}_P{P_MAX}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{exp_id}_P{P_MAX}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
    print("Saved:", out_pdf)


def main():
    # IMPORTANT: run on GPU node to match CUDA RNG used during training
    print("CUDA available:", torch.cuda.is_available())
    for e in EXPS:
        plot_one_exp(e["exp_id"], e["label"])


if __name__ == "__main__":
    main()