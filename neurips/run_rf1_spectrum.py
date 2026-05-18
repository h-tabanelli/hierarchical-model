#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import teacher
import estimators


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_range(s: str) -> list[int]:
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _parse_alphas(s: str) -> list[float]:
    s = s.strip()
    if ":" in s:
        a, b, h = s.split(":")
        a, b, h = float(a), float(b), float(h)
        out, k = [], 0
        while True:
            val = a + k * h
            if val > b + 1e-12:
                break
            out.append(round(val, 10))
            k += 1
        return out
    return [float(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Core computation (single alpha)
# ---------------------------------------------------------------------------

def _make_stream_fn_factory(*, d, p, n, batch_size, A_mode_teacher, beta, seed,
                             device, mean_y, std_y, g_name, B_mode, gamma):
    def stream_fn_factory():
        def stream_fn():
            for X_or_Z, _, y_norm, _, _ in teacher.stream_batches_teacher_y_normalized(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
                mean_y=mean_y, std_y=std_y, input_mode="true",
                g_name=g_name, g_callable=None, B_mode=B_mode, gamma=gamma,
            ):
                yield X_or_Z, y_norm
        return stream_fn
    return stream_fn_factory


@torch.no_grad()
def _run_one_alpha(
    *,
    d: int,
    p: int,
    alpha: float,
    seed: int,
    beta: float,
    batch_size: int,
    n_test: int,
    A_mode_teacher: str,
    B_mode: str,
    gamma: float,
    g_name: str,
    rf_width: int,
    rf_activation: str,
    n_iter_C_max: int,
    oversamp_C: int,
    device: torch.device,
    normalize_w: bool,
    fresh_proj: bool,
):
    t0 = time.time()
    n = max(int(round(d ** alpha)), 1)

    mean_y, std_y = teacher.compute_mean_std_y_stream(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=None, input_mode="true", B_mode=B_mode, gamma=gamma,
    )
    std_y = torch.clamp(std_y, min=1e-3)

    stream_kwargs = dict(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode_teacher=A_mode_teacher, beta=beta, device=device,
        mean_y=mean_y, std_y=std_y, g_name=g_name, B_mode=B_mode, gamma=gamma,
    )
    stream_fn_factory = _make_stream_fn_factory(seed=seed, **stream_kwargs)

    _FRESH_OFFSET = 1_000_003
    stream_fn_factory_fresh = (
        _make_stream_fn_factory(seed=seed + _FRESH_OFFSET, **stream_kwargs)
        if fresh_proj else stream_fn_factory
    )

    rf_layer, Vhat = estimators.fit_rf_spectral_layer1_from_stream(
        stream_fn_factory=stream_fn_factory,
        d=d, rf_width=rf_width, p_out=p, n_total=n,
        rf_activation=rf_activation, rf_seed=seed + 314159,
        n_iter=n_iter_C_max, oversamp=oversamp_C,
        device=device, Q_init=None, T_min=0, stop_tol=None,
        normalize_rows=normalize_w,
    )

    rng = np.random.default_rng(12345)
    Xte = torch.tensor(rng.normal(size=(n_test, d)).astype(np.float32), device=device)

    W = rf_layer["W"].to(device=device, dtype=torch.float32)
    U_dbg = Xte @ W.T
    S_dbg = estimators.apply_rf_layer(Xte, rf_layer=rf_layer, device=device, dtype=torch.float32)
    Vhat_f32 = Vhat.to(device=device, dtype=torch.float32)

    Z_dbg = (S_dbg @ Vhat_f32.T).detach()
    mu_z = Z_dbg.mean(dim=0, keepdim=True)
    Zc_dbg = Z_dbg - mu_z
    Cov_z = 0.5 * ((Zc_dbg.T @ Zc_dbg) / float(Z_dbg.shape[0]))
    Cov_z = 0.5 * (Cov_z + Cov_z.T)
    evals_z, evecs_z = torch.linalg.eigh(Cov_z)
    W_white = evecs_z @ torch.diag(torch.rsqrt(torch.clamp(evals_z, min=1e-8))) @ evecs_z.T
    H_dbg = Zc_dbg @ W_white

    u_mean = U_dbg.mean(dim=0).detach().cpu()
    u_var = U_dbg.var(dim=0, unbiased=False).detach().cpu()
    s_mean = S_dbg.mean(dim=0).detach().cpu()
    s_std = S_dbg.std(dim=0, unbiased=False).detach().cpu()
    s_var = S_dbg.var(dim=0, unbiased=False).detach().cpu()
    h_mean = H_dbg.mean(dim=0).detach().cpu()
    h_std = H_dbg.std(dim=0, unbiased=False).detach().cpu()
    h_var = H_dbg.var(dim=0, unbiased=False).detach().cpu()
    post_proj_mean = Z_dbg.mean(dim=0).cpu()
    post_proj_var = Z_dbg.var(dim=0, unbiased=False).cpu()

    Hc = H_dbg - H_dbg.mean(dim=0, keepdim=True)
    Cov_h = 0.5 * ((Hc.T @ Hc) / float(H_dbg.shape[0]))
    Cov_h = 0.5 * (Cov_h + Cov_h.T)
    cov_hhat_eigs = torch.linalg.eigvalsh(Cov_h).flip(0).detach().cpu()

    def rf_stream_train():
        for Xb, yb in stream_fn_factory()():
            Xb = Xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            Sb = estimators.apply_rf_layer(Xb, rf_layer=rf_layer, device=device, dtype=torch.float32)
            yield Sb, yb

    CQ_sel = estimators.C_apply_vec(rf_stream_train, Vhat, m=rf_width, n_total=n, device=device)
    Rsmall = Vhat.to(torch.float64) @ CQ_sel.to(torch.float64).T
    Rsmall = 0.5 * (Rsmall + Rsmall.T)
    selected_ritz_eigs = torch.linalg.eigvalsh(Rsmall).flip(0).detach().cpu()

    C_full = torch.zeros((rf_width, rf_width), dtype=torch.float64, device=device)
    for Xb, yb in stream_fn_factory_fresh()():
        Xb = Xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32).reshape(-1)
        Sb = estimators.apply_rf_layer(Xb, rf_layer=rf_layer, device=device, dtype=torch.float32)
        Sb64 = Sb.to(torch.float64)
        C_full += Sb64.T @ (Sb64 * yb[:, None].to(torch.float64))
    C_full /= float(n)
    C_full = 0.5 * (C_full + C_full.T)
    full_eigs = torch.linalg.eigvalsh(C_full).flip(0).detach().cpu()
    del C_full

    metrics = {
        "alpha": float(alpha),
        "n": int(n),
        "d": int(d),
        "p": int(p),
        "seed": int(seed),
        "beta": float(beta),
        "A_mode_teacher": str(A_mode_teacher),
        "B_mode": str(B_mode),
        "gamma": float(gamma),
        "g_name": str(g_name),
        "rf_width": int(rf_width),
        "rf_activation": str(rf_activation),
        "n_iter_C_max": int(n_iter_C_max),
        "oversamp_C": int(oversamp_C),
        "normalize_w": bool(normalize_w),
        "fresh_proj": bool(fresh_proj),
        "u_mean_avg": float(u_mean.mean()),
        "u_var_avg": float(u_var.mean()),
        "s_mean_rms": float(torch.linalg.norm(s_mean) / math.sqrt(len(s_mean))),
        "s_std_avg": float(s_std.mean()),
        "h_mean_l2": float(torch.linalg.norm(h_mean)),
        "h_std_avg": float(h_std.mean()),
        "wall_seconds": float(time.time() - t0),
        "device": str(device),
    }

    payload = {
        "metrics": metrics,
        "u_mean": u_mean,
        "u_var": u_var,
        "s_mean": s_mean,
        "s_var": s_var,
        "s_std": s_std,
        "h_mean": h_mean,
        "h_var": h_var,
        "h_std": h_std,
        "cov_hhat_eigs": cov_hhat_eigs,
        "selected_ritz_eigs": selected_ritz_eigs,
        "full_eigs": full_eigs,
        "Vhat": Vhat.detach().cpu(),
        "post_proj_mean": post_proj_mean,
        "post_proj_var": post_proj_var,
    }
    return metrics, payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="RF1 spectrum analysis: eigenspectrum of C in the random-feature basis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--exp_id", type=str, default="rf1_spectrum")
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.5,
                    help="Hidden dimension exponent: p = round(d^eps).")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alphas", type=str, required=True,
                    help='Alpha range "start:stop:step" or CSV.')
    ap.add_argument("--seeds", type=str, default="0",
                    help='Seeds: range "0-4" or CSV "0,1,2".')

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_test", type=int, default=3000)

    ap.add_argument("--A_mode_teacher", type=str, default="sym_orth_frob",
                    choices=["sym_orth_frob", "rank1_orth"])
    ap.add_argument("--B_mode", type=str, default="dense",
                    choices=["dense", "powerlaw_diag"])
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--g_name", type=str, default="id")

    ap.add_argument("--rf_width", type=int, default=8192,
                    help="Number of random features (= dimension of C matrix). "
                         "Reduce to e.g. 512 for quick tests.")
    ap.add_argument("--rf_activation", type=str, default="relu_l1")
    ap.add_argument("--n_iter_C_max", type=int, default=15)
    ap.add_argument("--oversamp_C", type=int, default=10)
    ap.add_argument("--normalize_w", action="store_true", default=False,
                    help="Row-normalize RF weight matrix W to unit sphere.")
    ap.add_argument("--fresh_proj", action="store_true", default=False,
                    help="Use a different seed for C_full formation to avoid train/eval overlap.")

    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    alphas = sorted(_parse_alphas(args.alphas))
    seeds = _parse_range(args.seeds)
    p = int(round(args.d ** args.eps))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"d={args.d} eps={args.eps} p={p} rf_width={args.rf_width} device={device}")
    print(f"alphas ({len(alphas)}): {alphas[0]:.3g} .. {alphas[-1]:.3g}  seeds: {seeds}")

    outroot = Path(args.outdir)

    for seed in seeds:
        _seed_everything(seed)
        seeddir = outroot / args.exp_id / f"seed={seed:04d}"
        seeddir.mkdir(parents=True, exist_ok=True)
        metrics_path = seeddir / "rf1_spectrum_metrics.jsonl"

        done: set[float] = set()
        if metrics_path.exists():
            with metrics_path.open("r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        done.add(float(rec["alpha"]))
                    except Exception:
                        pass

        with metrics_path.open("a") as f:
            for alpha in alphas:
                if float(alpha) in done:
                    print(f"  skip seed={seed} alpha={alpha:.4g} (already done)")
                    continue

                metrics, payload = _run_one_alpha(
                    d=args.d, p=p, alpha=alpha, seed=seed, beta=args.beta,
                    batch_size=args.batch_size, n_test=args.n_test,
                    A_mode_teacher=args.A_mode_teacher, B_mode=args.B_mode,
                    gamma=args.gamma, g_name=args.g_name,
                    rf_width=args.rf_width, rf_activation=args.rf_activation,
                    n_iter_C_max=args.n_iter_C_max, oversamp_C=args.oversamp_C,
                    device=device,
                    normalize_w=args.normalize_w, fresh_proj=args.fresh_proj,
                )

                alpha_tag = f"{alpha:.4f}"
                torch.save(payload, seeddir / f"rf1_spectrum_alpha={alpha_tag}.pt")
                f.write(json.dumps(metrics) + "\n")
                f.flush()

                print(
                    f"  seed={seed:04d} alpha={alpha:.4g} n={metrics['n']} "
                    f"s_mean_rms={metrics['s_mean_rms']:.4g} "
                    f"s_std_avg={metrics['s_std_avg']:.4g} "
                    f"wall={metrics['wall_seconds']:.1f}s"
                )


if __name__ == "__main__":
    main()
