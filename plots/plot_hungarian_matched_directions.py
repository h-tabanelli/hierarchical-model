from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- make repo root importable (teacher.py at repo root) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import teacher  # noqa


# =========================
# CONFIG
# =========================

MODEL = "true"
P_MAX = 16                  # number of teacher directions to plot (<= p)
A_MODE_TEACHER = "sym_orth_frob"

RESULTS_ROOT = Path("results")
OUTDIR = Path("figures/hungarian_matching")
OUTNAME = "matched_corr_per_teacher_direction_vs_alpha"

EXPS = [
    {"label": r"$\gamma=0.4$", "exp_id": "D400_eps05_g04_v2"},
    {"label": r"$\gamma=1.0$", "exp_id": "D400_eps05_g10_v2"},
]

LINEWIDTH = 1.6
BAND_ALPHA = 0.15
GRID_ALPHA = 0.


# =========================
# Helpers
# =========================

def _round_alpha(a: float) -> float:
    return float(np.round(a, 10))

def _iter_estimates(exp_id: str):
    base = RESULTS_ROOT / exp_id
    yield from base.rglob("estimates.pt")

@torch.no_grad()
def build_Atrue_sym_orth_frob(d: int, p: int, seed: int) -> torch.Tensor:
    """
    IMPORTANT: use CUDA generator if available to match GPU-generated teachers.
    Return Atrue on CPU.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher["A"].detach().to("cpu")

def frob_normalized_score_matrix(Ahat: torch.Tensor, Atrue: torch.Tensor) -> np.ndarray:
    """
    Ahat, Atrue: (p,d,d) on CPU.
    Return score matrix S_{i,j} = |<Ahat_i, Atrue_j>| / (||Ahat_i|| ||Atrue_j||).
    """
    Ah = Ahat.to(torch.float32).reshape(Ahat.shape[0], -1)   # (p, m)
    At = Atrue.to(torch.float32).reshape(Atrue.shape[0], -1) # (p, m)

    # norms
    nh = torch.linalg.norm(Ah, dim=1).clamp_min(1e-12)  # (p,)
    nt = torch.linalg.norm(At, dim=1).clamp_min(1e-12)  # (p,)

    # correlation matrix
    C = (Ah @ At.T) / (nh[:, None] * nt[None, :])        # (p,p)
    S = torch.abs(C).cpu().numpy()                       # (p,p) in [0,1] approx
    return S

def hungarian_match_scores(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given score matrix S (p,p), find assignment maximizing sum S[i,j].
    Returns:
      row_ind (hat indices), col_ind (teacher indices) of matched pairs.
    """
    row_ind, col_ind = linear_sum_assignment(-S)  # maximize
    return row_ind, col_ind

def sem(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))

def collect_matched_table(exp_id: str) -> pd.DataFrame:
    """
    Output tidy table with:
      alpha, seed, j (teacher direction 1..P_MAX), score
    where score = matched correlation for teacher j.
    """
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
                raise ValueError("This script assumes A_MODE_TEACHER='sym_orth_frob'.")
            Atrue_cache[key] = build_Atrue_sym_orth_frob(d=d, p=p, seed=seed)

        Atrue = Atrue_cache[key]

        # scores + matching
        S = frob_normalized_score_matrix(Ahat, Atrue)    # (p,p)
        hi, tj = hungarian_match_scores(S)

        # Build teacher->hat mapping
        teacher_to_hat = np.empty(p, dtype=int)
        teacher_to_hat[tj] = hi

        # Collect first P_MAX teacher directions
        m = min(int(P_MAX), int(p))
        for j in range(m):
            i = teacher_to_hat[j]
            rows.append({
                "alpha": alpha,
                "seed": seed,
                "j": j + 1,                 # 1-indexed for plotting
                "score": float(S[i, j]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No matched rows for exp_id={exp_id}. "
            f"Check results/{exp_id}/.../estimates.pt exists and MODEL matches."
        )
    return df

def plot_one_exp(exp_id: str, label: str):
    df = collect_matched_table(exp_id)

    # aggregate over seeds
    g = (
        df.groupby(["alpha", "j"])["score"]
          .agg(["mean", sem, "count"])
          .reset_index()
          .rename(columns={"mean": "score_mean", "sem": "score_sem"})
          .sort_values(["j", "alpha"])
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.4), constrained_layout=True)
    ax.set_title(f"{label}  (hungarian matching, exp_id={exp_id})", fontsize=13)

    for j in sorted(g["j"].unique()):
        gj = g[g["j"] == j]
        x = gj["alpha"].to_numpy(float)
        y = gj["score_mean"].to_numpy(float)
        e = gj["score_sem"].to_numpy(float)

        line, = ax.plot(x, y, linewidth=LINEWIDTH, label=f"teacher j={int(j)}")
        col = line.get_color()
        if np.all(np.isfinite(e)):
            ax.fill_between(x, y - e, y + e, alpha=BAND_ALPHA, color=col)

    ax.set_xlabel(r"$\alpha=\log(n)/\log(d)$")
    ax.set_ylabel(r"matched corr  $|\langle \hat A_i, A^\star_j\rangle_F| / (\|\hat A_i\|_F\|A^\star_j\|_F)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(ncol=2, fontsize=8.5, frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))

    out_png = OUTDIR / f"{OUTNAME}_{exp_id}_P{P_MAX}.png"
    out_pdf = OUTDIR / f"{OUTNAME}_{exp_id}_P{P_MAX}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
    print("Saved:", out_pdf)

def main():
    print("CUDA available:", torch.cuda.is_available())
    for e in EXPS:
        plot_one_exp(e["exp_id"], e["label"])

if __name__ == "__main__":
    main()