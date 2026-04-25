from __future__ import annotations

from pathlib import Path
import json
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
EXP_ID = 'D800_eps05_g10_id_true_cal'
RESULTS_ROOT = Path('results')
OUTDIR = Path('figures/powerlaw_tail')
OUTNAME = 'tail_vs_mse_d800_g10'

# filters
MODEL = 'true'
G_NAME = 'id'
B_MODE = 'powerlaw_diag'
HEAD_MODE = 'spectral_B'
LAYER1_MODE = 'hermite_spectral'
CALIBRATE_OUTPUT = True
D = 800
EPS = 0.5
GAMMA = 1.0
A_MODE_TEACHER = 'sym_orth_frob'
METRIC = 'mse'      # 'mse', 'nmse', 'mse_scaled', 'nmse_scaled'

# theory / surrogate
K_LAYER = 2.0       # first-direction threshold baseline: alpha_1 ~= K_LAYER
USE_CGAMMA_SHIFT = False  # if True, include + 2 log(C_gamma)/log(d) in alpha_j with C_gamma = 1/sqrt(sum_j j^{-2gamma})
SQUARE_MATCH_SCORE = True # q_j = matched_score^2
NORMALIZE_TAILS = False   # divide R_num / R_th by sum_j w_j
ALIGN_TO_MSE = True       # vertical alignment by constant factor on a chosen window
ALIGN_X0 = 2.45
ALIGN_X1 = 3.00

# plotting
X_LIM = (0.5, 3.1)
Y_LIM_LEFT = None
DRAW_SEED_LINES = True
DRAW_RATE = False         # optional legacy asymptotic line on MSE panel
RATE_X0 = 2.55
RATE_X1 = 3.00
RATE_EXP_N = -1.0 + 1.0/(2.0*GAMMA)   # e.g. gamma=1 -> -1/2
BAND_ALPHA = 0.16
GRID_ALPHA = 0.18
FIGSIZE = (12.5, 4.6)
MARKER_SIZE = 5.0

REPO_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path('.').resolve()
sys.path.insert(0, str(REPO_ROOT))
import teacher  # noqa: E402


# =========================
# helpers
# =========================
def _round_alpha(a: float) -> float:
    return float(np.round(float(a), 10))


def _maybe_filter(df: pd.DataFrame, col: str, value: Any) -> pd.DataFrame:
    if col not in df.columns or value is None:
        return df
    s = df[col]
    if isinstance(value, bool):
        if pd.api.types.is_bool_dtype(s):
            return df[s == value]
        if pd.api.types.is_numeric_dtype(s):
            return df[s.astype(int) == int(value)]
        return df[s.astype(str).str.lower() == str(value).lower()]
    if isinstance(value, (int, float)) and pd.api.types.is_numeric_dtype(s):
        return df[np.isclose(s.astype(float), float(value))]
    return df[s.astype(str) == str(value)]


def _iter_metrics(exp_id: str):
    base = RESULTS_ROOT / exp_id
    for pth in base.rglob('metrics.jsonl'):
        jobdir = pth.parent
        with pth.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row['_jobdir'] = str(jobdir)
                yield row


def _estimates_path(row: dict[str, Any]) -> Path:
    jobdir = Path(row['_jobdir'])
    alpha = float(row['alpha'])
    model = str(row['model'])
    head = str(row['head_mode'])
    return jobdir / 'artifacts' / f'alpha={alpha:.4f}' / f'model={model}' / f'head={head}' / 'estimates.pt'


@torch.no_grad()
def build_Atrue_sym_orth_frob(d: int, p: int, seed: int) -> torch.Tensor:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, dev)
    return A_teacher['A'].detach().to('cpu')


def frob_normalized_score_matrix(Ahat: torch.Tensor, Atrue: torch.Tensor) -> np.ndarray:
    Ah = Ahat.to(torch.float32).reshape(Ahat.shape[0], -1)
    At = Atrue.to(torch.float32).reshape(Atrue.shape[0], -1)
    nh = torch.linalg.norm(Ah, dim=1).clamp_min(1e-12)
    nt = torch.linalg.norm(At, dim=1).clamp_min(1e-12)
    C = (Ah @ At.T) / (nh[:, None] * nt[None, :])
    return torch.abs(C).cpu().numpy()


def matched_teacher_scores(Ahat: torch.Tensor, Atrue: torch.Tensor) -> np.ndarray:
    S = frob_normalized_score_matrix(Ahat, Atrue)
    row_ind, col_ind = linear_sum_assignment(-S)
    teacher_to_hat = np.empty(S.shape[1], dtype=int)
    teacher_to_hat[col_ind] = row_ind
    scores = S[teacher_to_hat, np.arange(S.shape[1])]
    return scores


def sem_from_values(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def c_gamma_inverse_sqrt_sum(p: int, gamma: float) -> float:
    idx = np.arange(1, p + 1, dtype=float)
    s = np.sum(idx ** (-2.0 * gamma))
    return float(1.0 / np.sqrt(s + 1e-300))


def recovered_count_theory(alpha: float, d: int, p: int, gamma: float, alpha0: float, use_cgamma_shift: bool) -> int:
    # alpha_j ~= alpha0 + (2 gamma log j - 2 log C_gamma)/log d
    shift = 0.0
    if use_cgamma_shift:
        Cg = c_gamma_inverse_sqrt_sum(p, gamma)
        shift = (-2.0 * np.log(Cg)) / np.log(d)
    if gamma <= 0:
        return int(np.clip(p if alpha >= alpha0 + shift else 0, 0, p))
    val = np.exp(((alpha - alpha0 - shift) * np.log(d)) / (2.0 * gamma))
    return int(np.clip(np.floor(val + 1e-12), 0, p))


def align_factor(x: np.ndarray, y_ref: np.ndarray, y_cmp: np.ndarray, x0: float, x1: float) -> float:
    mask = (x >= x0) & (x <= x1) & np.isfinite(y_ref) & np.isfinite(y_cmp) & (y_ref > 0) & (y_cmp > 0)
    if not np.any(mask):
        return 1.0
    return float(np.exp(np.mean(np.log(y_ref[mask]) - np.log(y_cmp[mask]))))


# =========================
# main
# =========================
def main():
    rows = list(_iter_metrics(EXP_ID))
    if not rows:
        raise RuntimeError(f'No metrics.jsonl found under {RESULTS_ROOT / EXP_ID}')
    df = pd.DataFrame(rows)

    for col, val in [
        ('model', MODEL),
        ('g_name', G_NAME),
        ('B_mode', B_MODE),
        ('head_mode', HEAD_MODE),
        ('layer1_mode', LAYER1_MODE),
        ('calibrate_output', CALIBRATE_OUTPUT),
        ('d', D),
        ('eps', EPS),
        ('gamma', GAMMA),
    ]:
        df = _maybe_filter(df, col, val)

    if df.empty:
        raise RuntimeError('No metrics rows left after filtering. Check CONFIG.')

    # one row per (alpha, seed) expected
    df = df.sort_values(['seed', 'alpha']).copy()
    if METRIC not in df.columns:
        raise KeyError(f'{METRIC=} not present in metrics rows.')
    if 'ovH' not in df.columns:
        raise KeyError('ovH not present in metrics rows.')

    # reconstruct matched per-teacher overlaps and R_num / R_th for each row
    Atrue_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    r_rows = []
    detail_rows = []

    for _, row in df.iterrows():
        est_path = _estimates_path(row)
        if not est_path.exists():
            # skip rows without artifacts
            continue
        obj = torch.load(est_path, map_location='cpu')
        Ahat = obj.get('Ahat', None)
        if Ahat is None:
            continue
        d = int(row['d'])
        p = int(row['p'])
        seed = int(row['seed'])
        alpha = float(row['alpha'])

        key = (d, p, seed)
        if key not in Atrue_cache:
            if A_MODE_TEACHER != 'sym_orth_frob':
                raise ValueError('This script assumes A_MODE_TEACHER="sym_orth_frob".')
            Atrue_cache[key] = build_Atrue_sym_orth_frob(d=d, p=p, seed=seed)
        Atrue = Atrue_cache[key]

        scores = matched_teacher_scores(Ahat.to('cpu'), Atrue)  # teacher-indexed, in [0,1]
        q = scores ** 2 if SQUARE_MATCH_SCORE else scores
        j = np.arange(1, p + 1, dtype=float)
        w = j ** (-2.0 * GAMMA)
        wsum = float(np.sum(w))
        r_num = float(np.sum(w * (1.0 - q)))
        m_rec = recovered_count_theory(alpha=alpha, d=d, p=p, gamma=GAMMA, alpha0=K_LAYER, use_cgamma_shift=USE_CGAMMA_SHIFT)
        tail_mask = (j > float(m_rec))
        r_th = float(np.sum(w[tail_mask]))
        if NORMALIZE_TAILS:
            r_num /= (wsum + 1e-300)
            r_th /= (wsum + 1e-300)

        r_rows.append({
            'alpha': alpha,
            'seed': seed,
            'R_num': r_num,
            'R_th': r_th,
            'm_theory': m_rec,
        })
        for jj, (qq, ww) in enumerate(zip(q, w), start=1):
            detail_rows.append({
                'alpha': alpha,
                'seed': seed,
                'j': jj,
                'q_match': float(qq),
                'w': float(ww),
                'w_gap': float(ww * (1.0 - qq)),
            })

    rdf = pd.DataFrame(r_rows)
    if rdf.empty:
        raise RuntimeError('No artifacts/estimates.pt found after filtering. Did you run with save_estimates=True?')

    # merge numerical surrogates back to metrics
    df2 = df.merge(rdf, on=['alpha', 'seed'], how='inner')
    if df2.empty:
        raise RuntimeError('Merge between metrics rows and tail rows is empty.')

    # aggregate by alpha
    g = (
        df2.groupby('alpha')[[METRIC, 'ovH', 'R_num', 'R_th', 'm_theory']]
           .agg(['mean', 'std', 'count'])
           .reset_index()
    )
    g.columns = ['alpha' if c[0] == 'alpha' else f'{c[0]}_{c[1]}' for c in g.columns.to_flat_index()]
    g = g.sort_values('alpha').copy()

    x = g['alpha'].to_numpy(float)
    y_mse = g[f'{METRIC}_mean'].to_numpy(float)
    y_mse_sem = np.divide(g[f'{METRIC}_std'].to_numpy(float), np.sqrt(np.maximum(g[f'{METRIC}_count'].to_numpy(float), 1.0)), where=np.isfinite(g[f'{METRIC}_std'].to_numpy(float)))
    y_ov = g['ovH_mean'].to_numpy(float)
    y_ov_sem = np.divide(g['ovH_std'].to_numpy(float), np.sqrt(np.maximum(g['ovH_count'].to_numpy(float), 1.0)), where=np.isfinite(g['ovH_std'].to_numpy(float)))
    y_rn = g['R_num_mean'].to_numpy(float)
    y_rn_sem = np.divide(g['R_num_std'].to_numpy(float), np.sqrt(np.maximum(g['R_num_count'].to_numpy(float), 1.0)), where=np.isfinite(g['R_num_std'].to_numpy(float)))
    y_rt = g['R_th_mean'].to_numpy(float)
    y_mth = g['m_theory_mean'].to_numpy(float)

    if ALIGN_TO_MSE:
        c_num = align_factor(x, y_mse, y_rn, ALIGN_X0, ALIGN_X1)
        c_th = align_factor(x, y_mse, y_rt, ALIGN_X0, ALIGN_X1)
    else:
        c_num, c_th = 1.0, 1.0

    y_rn_plot = c_num * y_rn
    y_rn_sem_plot = c_num * y_rn_sem
    y_rt_plot = c_th * y_rt

    # plotting
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    fig.suptitle(f'd={D}, eps={EPS}, gamma={GAMMA}, model={MODEL}, calib={CALIBRATE_OUTPUT}', fontsize=15)

    # left: MSE + aligned numerical and discrete theoretical tails
    if DRAW_SEED_LINES:
        for seed, gs in df2.groupby('seed'):
            gs = gs.sort_values('alpha')
            ax1.plot(gs['alpha'], gs[METRIC], color='tab:blue', alpha=0.22, lw=1.0)
            ax1.plot(gs['alpha'], c_num * gs['R_num'], color='tab:orange', alpha=0.18, lw=1.0)

    line_mse, = ax1.plot(x, y_mse, marker='o', ms=MARKER_SIZE, lw=2.2, color='tab:green', label=METRIC.upper())
    lo = np.maximum(y_mse - y_mse_sem, 1e-16)
    hi = np.maximum(y_mse + y_mse_sem, lo * (1 + 1e-12))
    ax1.fill_between(x, lo, hi, color=line_mse.get_color(), alpha=BAND_ALPHA, linewidth=0)

    line_rn, = ax1.plot(x, y_rn_plot, marker='s', ms=4.2, lw=2.0, color='tab:orange', label=r'$c_{\rm num} R_{\rm num}(\alpha)$')
    lo_rn = np.maximum(y_rn_plot - y_rn_sem_plot, 1e-16)
    hi_rn = np.maximum(y_rn_plot + y_rn_sem_plot, lo_rn * (1 + 1e-12))
    ax1.fill_between(x, lo_rn, hi_rn, color=line_rn.get_color(), alpha=0.12, linewidth=0)

    ax1.plot(x, y_rt_plot, lw=2.5, color='tab:red', linestyle='--', label=r'$c_{\rm th} R_{\rm th}(\alpha)$')

    if DRAW_RATE:
        mask = (x >= RATE_X0) & (x <= RATE_X1) & np.isfinite(y_mse) & (y_mse > 0)
        if np.any(mask):
            mid = 0.5 * (RATE_X0 + RATE_X1)
            y_mid = float(np.interp(mid, x[mask], y_mse[mask]))
            xs = np.array([RATE_X0, RATE_X1], dtype=float)
            ys = y_mid * (float(D) ** (RATE_EXP_N * (xs - mid)))
            ax1.plot(xs, ys, color='black', lw=4.0, solid_capstyle='round')

    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\alpha = \log(n)/\log(d)$')
    ax1.set_ylabel(METRIC.upper())
    ax1.set_xlim(*X_LIM)
    if Y_LIM_LEFT is not None:
        ax1.set_ylim(*Y_LIM_LEFT)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(frameon=True, fontsize=10)

    # right: ovH and recovered-count theory
    line_ov, = ax2.plot(x, y_ov, marker='o', ms=MARKER_SIZE, lw=2.2, color='tab:green', label='ovH')
    ax2.fill_between(
        x,
        np.clip(y_ov - y_ov_sem, 0.0, 1.0),
        np.clip(y_ov + y_ov_sem, 0.0, 1.0),
        color=line_ov.get_color(), alpha=BAND_ALPHA, linewidth=0,
    )
    ax2.set_xlabel(r'$\alpha = \log(n)/\log(d)$')
    ax2.set_ylabel('ovH')
    ax2.set_xlim(*X_LIM)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=GRID_ALPHA)

    ax2b = ax2.twinx()
    ax2b.plot(x, y_mth, color='tab:red', linestyle='--', lw=2.2, label=r'$m_{\rm th}(\alpha)$')
    ax2b.set_ylabel(r'theoretical recovered count $m_{\rm th}(\alpha)$')
    ax2b.set_ylim(0, max(float(df2['p'].iloc[0]), np.nanmax(y_mth) + 1.0))

    # combined legend on right
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=True, fontsize=10, loc='upper left')

    out_png = OUTDIR / f'{OUTNAME}.png'
    out_pdf = OUTDIR / f'{OUTNAME}.pdf'
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print('Saved:', out_png)
    print('Saved:', out_pdf)

    # save underlying aggregated data too
    out_csv = OUTDIR / f'{OUTNAME}_aggregated.csv'
    g.assign(R_num_aligned=y_rn_plot, R_th_aligned=y_rt_plot).to_csv(out_csv, index=False)
    print('Saved:', out_csv)

    out_detail = OUTDIR / f'{OUTNAME}_detail_per_direction.csv'
    pd.DataFrame(detail_rows).to_csv(out_detail, index=False)
    print('Saved:', out_detail)


if __name__ == '__main__':
    main()
