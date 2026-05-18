#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import teacher
import estimators


def _read_task(taskfile: Path, task_id: int) -> dict:
    with taskfile.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id={task_id} out of range for {taskfile}")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_stream_fn_factory(*, d, p, n, batch_size, A_mode_teacher, beta, seed,
                             device, mean_y, std_y, g_name, B_mode, gamma):
    """Return a stream_fn_factory that replays the teacher stream with the given seed."""
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

    n = int(round(d ** alpha))
    n = max(n, 1)

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

    # training stream (used to fit Vhat)
    stream_fn_factory = _make_stream_fn_factory(seed=seed, **stream_kwargs)

    # fresh stream for C_full (different seed to avoid train/eval overlap)
    _FRESH_OFFSET = 1_000_003  # prime
    stream_fn_factory_fresh = (
        _make_stream_fn_factory(seed=seed + _FRESH_OFFSET, **stream_kwargs)
        if fresh_proj else stream_fn_factory
    )

    rf_seed_eff = seed + 314159

    rf_layer, Vhat = estimators.fit_rf_spectral_layer1_from_stream(
        stream_fn_factory=stream_fn_factory,
        d=d, rf_width=rf_width, p_out=p, n_total=n,
        rf_activation=rf_activation, rf_seed=rf_seed_eff,
        n_iter=n_iter_C_max, oversamp=oversamp_C,
        device=device, Q_init=None, T_min=0, stop_tol=None,
        normalize_rows=normalize_w,
    )

    # --- test-set statistics on n_test fresh points ---
    rng = np.random.default_rng(12345)
    Xte = torch.tensor(rng.normal(size=(n_test, d)).astype(np.float32), device=device)

    W = rf_layer["W"].to(device=device, dtype=torch.float32)
    U_dbg = Xte @ W.T
    S_dbg = estimators.apply_rf_layer(Xte, rf_layer=rf_layer, device=device, dtype=torch.float32)
    Vhat_f32 = Vhat.to(device=device, dtype=torch.float32)

    Z_dbg = (S_dbg @ Vhat_f32.T).detach()                   # (n_test, p)

    mu_z = Z_dbg.mean(dim=0, keepdim=True)
    Zc_dbg = Z_dbg - mu_z
    Cov_z = (Zc_dbg.T @ Zc_dbg) / float(Z_dbg.shape[0])
    Cov_z = 0.5 * (Cov_z + Cov_z.T)
    evals_z, evecs_z = torch.linalg.eigh(Cov_z)
    W_white = evecs_z @ torch.diag(torch.rsqrt(torch.clamp(evals_z, min=1e-8))) @ evecs_z.T
    H_dbg = Zc_dbg @ W_white

    post_proj_mean = Z_dbg.mean(dim=0).cpu()
    post_proj_var  = Z_dbg.var(dim=0, unbiased=False).cpu()

    u_mean = U_dbg.mean(dim=0).detach().cpu()
    u_var  = U_dbg.var(dim=0, unbiased=False).detach().cpu()

    s_mean = S_dbg.mean(dim=0).detach().cpu()
    s_std  = S_dbg.std(dim=0, unbiased=False).detach().cpu()
    s_var  = S_dbg.var(dim=0, unbiased=False).detach().cpu()

    h_mean = H_dbg.mean(dim=0).detach().cpu()
    h_std  = H_dbg.std(dim=0, unbiased=False).detach().cpu()
    h_var  = H_dbg.var(dim=0, unbiased=False).detach().cpu()

    Hc = H_dbg - H_dbg.mean(dim=0, keepdim=True)
    Cov_h = (Hc.T @ Hc) / float(H_dbg.shape[0])
    Cov_h = 0.5 * (Cov_h + Cov_h.T)
    cov_hhat_eigs = torch.linalg.eigvalsh(Cov_h).flip(0).detach().cpu()

    # --- Ritz eigenvalues on the learned subspace (training stream) ---
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

    # --- Full spectrum via dense C_full (on GPU for speed) ---
    # Use fresh stream when fresh_proj=True to avoid train/eval overlap.
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
    del C_full  # free GPU memory

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
        "spec_rank": int(rf_width),
        "u_mean_avg": float(u_mean.mean()),
        "u_mean_min": float(u_mean.min()),
        "u_mean_max": float(u_mean.max()),
        "u_var_avg": float(u_var.mean()),
        "u_var_min": float(u_var.min()),
        "u_var_max": float(u_var.max()),
        "s_mean_l2": float(torch.linalg.norm(s_mean)),
        "s_mean_rms": float(torch.linalg.norm(s_mean) / math.sqrt(len(s_mean))),
        "s_std_avg": float(s_std.mean()),
        "s_std_min": float(s_std.min()),
        "s_std_max": float(s_std.max()),
        "h_mean_l2": float(torch.linalg.norm(h_mean)),
        "h_std_avg": float(h_std.mean()),
        "h_std_min": float(h_std.min()),
        "h_std_max": float(h_std.max()),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskfile", type=str, required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--spec_rank", type=int, default=256)
    args = ap.parse_args()

    task = _read_task(Path(args.taskfile), args.task_id)

    d = int(task["d"])
    eps = float(task["eps"])
    p = int(round(d ** eps))

    beta = float(task.get("beta", 1.0))
    batch_size = int(task.get("batch_size", 2048))
    n_test = int(task.get("n_test", 3000))

    A_mode_teacher = str(task.get("A_mode_teacher", "sym_orth_frob"))
    B_mode = str(task.get("B_mode", "dense"))
    gamma = float(task.get("gamma", 0.0))
    g_name = str(task.get("g_name", "id"))

    rf_width = int(task.get("rf_width", 8192))
    rf_activation = str(task.get("rf_activation", "relu_l1"))
    n_iter_C_max = int(task.get("n_iter_C_max", 15))
    oversamp_C = int(task.get("oversamp_C", 10))

    normalize_w = bool(task.get("normalize_w", False))
    fresh_proj = bool(task.get("fresh_proj", False))

    exp_id = str(task.get("exp_id", "rf1_spectrum"))
    chunk_id = int(task.get("chunk_id", args.task_id))
    seed = int(task["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(seed)

    outroot = Path(args.outdir) / exp_id / f"chunk={chunk_id:04d}" / f"seed={seed:04d}"
    outroot.mkdir(parents=True, exist_ok=True)

    metrics_path = outroot / "rf1_spectrum_metrics.jsonl"

    with metrics_path.open("a", encoding="utf-8") as f:
        for alpha in [float(a) for a in task["alphas"]]:
            metrics, payload = _run_one_alpha(
                d=d, p=p, alpha=alpha, seed=seed, beta=beta,
                batch_size=batch_size, n_test=n_test,
                A_mode_teacher=A_mode_teacher, B_mode=B_mode,
                gamma=gamma, g_name=g_name,
                rf_width=rf_width, rf_activation=rf_activation,
                n_iter_C_max=n_iter_C_max, oversamp_C=oversamp_C,
                device=device,
                normalize_w=normalize_w, fresh_proj=fresh_proj,
            )

            alpha_tag = f"{alpha:.4f}"
            torch.save(payload, outroot / f"rf1_spectrum_alpha={alpha_tag}.pt")
            f.write(json.dumps(metrics) + "\n")
            f.flush()

            print(
                f"chunk={chunk_id:04d} seed={seed:04d} alpha={alpha:.4g} "
                f"n={metrics['n']} s_mean_rms={metrics['s_mean_rms']:.4g} "
                f"s_std_avg={metrics['s_std_avg']:.4g} h_mean_l2={metrics['h_mean_l2']:.4g}"
            )


if __name__ == "__main__":
    main()
