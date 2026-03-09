#!/usr/bin/env python3
"""Run ONE *chunk* task for the 2-layer model and write results incrementally.

Designed for SLURM job arrays.

Outputs (append-safe):
  results/<exp_id>/chunk=<chunk_id>/seed=<seed>/metrics.jsonl
  results/<exp_id>/chunk=<chunk_id>/seed=<seed>/task_meta.json
  Q_last_<model>.pt  (warm-start checkpoints)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import teacher
import estimators
import measures as mps

from cluster_tools.utils_seeding import seed_everything


def _default_device(device_str: str | None):
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_done_pairs(metrics_path: Path) -> set[tuple[float, str]]:
    done: set[tuple[float, str]] = set()
    if not metrics_path.exists():
        return done
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((float(rec["alpha"]), str(rec["model"])))
            except Exception:
                continue
    return done


def _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model: str, device):
    """Same computation as in run_2layers_nmse.py."""
    if model == "true":
        Hhat = estimators.compute_hhat_from_X_and_Ahat(
            X_or_Z.to(device=device, dtype=torch.float32), Ahat
        ).to(device=device, dtype=torch.float32)
    else:
        Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
        Hhat = X_or_Z.to(device=device, dtype=torch.float32) @ Ahat_flat.T

    Bhat_t = Bhat_cpu.to(device=device, dtype=torch.float32)
    trB = torch.trace(Bhat_t)
    s = torch.einsum("bp,pq,bq->b", Hhat, Bhat_t, Hhat) - trB
    return s


@torch.no_grad()
def _run_one_alpha_model(
    *,
    d: int,
    p: int,
    alpha: float,
    seed: int,
    model: str,
    beta: float,
    n_test: int,
    batch_size: int,
    A_mode_teacher: str,
    B_mode: str,
    gamma: float,
    g_name: str,
    fit_degree: int,
    fit_ridge: float,
    n_iter_C_max: int,
    oversamp_C: int,
    T_min: int,
    stop_tol: float | None,
    Q_init: torch.Tensor | None,
    device: torch.device,
    Xte: torch.Tensor,
) -> tuple[dict, torch.Tensor | None]:
    """Run a single (alpha, model) and return (metrics_dict, Q_full_for_warm_start)."""
    t0 = time.time()

    n = int(round(d ** float(alpha)))
    n = max(n, 1)

    act = teacher.get_activation_fn(g_name=g_name, g_callable=None)
    is_id = (g_name is None) or (g_name == "id")

    # ---- pass 1: mean/std ----
    mean_y, std_y = teacher.compute_mean_std_y_stream(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=None, input_mode=model,
        B_mode=B_mode, gamma=gamma
    )
    std_y = torch.clamp(std_y, min=1e-3)

    # ---- stream factory (replay normalized y) ----
    def stream_fn_factory():
        def stream_fn():
            for X_or_Z, _, y_norm, _, _ in teacher.stream_batches_teacher_y_normalized(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
                mean_y=mean_y, std_y=std_y, input_mode=model,
                g_name=g_name, g_callable=None,
                B_mode=B_mode, gamma=gamma
            ):
                yield X_or_Z, y_norm
        return stream_fn

    # ---- Step 1: Ahat (warm start + early stop on iterations) ----
    if stop_tol is None:
        Ahat = estimators.top_p_eigmats_of_C(
            stream_fn_factory=stream_fn_factory,
            d=d, n_total=n, p=p,
            n_iter=n_iter_C_max,
            oversamp=oversamp_C,
            device=device,
            input_mode=model,
            Q_init=Q_init
        )
        Q_full = None
        info = {"n_iter_effective": n_iter_C_max, "stop_delta": None}
    else:
        Ahat, Q_full, info = estimators.top_p_eigmats_of_C(
            stream_fn_factory=stream_fn_factory,
            d=d, n_total=n, p=p,
            n_iter=n_iter_C_max,
            oversamp=oversamp_C,
            device=device,
            input_mode=model,
            Q_init=Q_init,
            T_min=T_min,
            stop_tol=stop_tol,
            return_Q_full=True,
            return_info=True
        )

    # ---- Step 2: Bhat ----
    if model == "true":
        Bhat_cpu = estimators.estimate_Bhat_from_stream(
            stream_fn=stream_fn_factory(),   # IMPORTANT: pass callable, not generator
            Ahat=Ahat, p=p, n_total=n,
            device=device
        )
    else:
        Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)

        def Hhat_stream():
            for Z, y_norm in stream_fn_factory()():
                Z = Z.to(device=device, dtype=torch.float32)
                Hhat = Z @ Ahat_flat.T
                yield Hhat, y_norm

        Bhat_cpu = estimators.estimate_Bhat_from_H_stream(
            stream_fn=Hhat_stream, p=p, n_total=n, device=device
        )

    # ---- calibration / link ----
    if is_id:
        s_list = []
        for j, (X_or_Z, _) in enumerate(stream_fn_factory()()):
            s = _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
            s_list.append(s.detach().cpu().numpy())
            if j + 1 >= 2:
                break
        s_cal = np.concatenate(s_list, axis=0)
        scale = 1.0 / (np.std(s_cal) + 1e-12)
        Bhat_cpu = Bhat_cpu * float(scale)
        coeffs = None
        mu_sig = None
    else:
        s_list, y_list = [], []
        for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
            s = _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
            s_list.append(s.detach().cpu().numpy())
            y_list.append(y_norm.detach().cpu().numpy())
            if j + 1 >= 10:
                break
        s_fit = np.concatenate(s_list, axis=0)
        y_fit = np.concatenate(y_list, axis=0)
        coeffs, mu_sig = estimators.fit_polynomial_link(s_fit, y_fit, degree=fit_degree, ridge=fit_ridge)

    # ---- rebuild teacher (for overlaps + test y) ----
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if A_mode_teacher == "rank1_orth":
        A_teacher = teacher.gen_A_rank1_orth_torch(d, p, gen, device)
    else:
        A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, device)

    if B_mode == "dense":
        B_teacher = teacher.gen_B_symmetric_dense_torch(p, gen, device, beta=beta)
    elif B_mode == "powerlaw_diag":
        B_teacher = teacher.gen_B_powerlaw_diag_torch(p, gen, device, beta=beta, gamma=gamma, rademacher=True)
    else:
        raise ValueError(f"Unknown B_mode: {B_mode}")

    ovA = None
    if A_mode_teacher == "sym_orth_frob":
        Atrue = A_teacher["A"].detach()
        ovA = float(mps.subspace_overlap_frob(Ahat.detach(), Atrue))

    if model == "true":
        Hte_true = teacher.compute_h_from_X_torch(Xte, A_teacher).to(device=device, dtype=torch.float32)
        Hte_hat = estimators.compute_hhat_from_X_and_Ahat(Xte, Ahat).to(device=device, dtype=torch.float32)
        Hte_true_for_overlap = Hte_true
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
        ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        X_or_Z_test = Xte
    else:
        Ahat_flat  = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
        Atrue_flat = teacher.flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device=device, dtype=torch.float32)
        gen_te = torch.Generator(device=device); gen_te.manual_seed(seed + 9999)
        Zte = torch.randn(n_test, Ahat_flat.shape[1], generator=gen_te, device=device, dtype=torch.float32)
        Hte_hat = Zte @ Ahat_flat.T
        Hte_true_for_overlap = Zte @ Atrue_flat.T
        Hte_true = Hte_true_for_overlap
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
        ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        X_or_Z_test = Zte

    s_te = teacher.compute_y_from_H_torch(Hte_true, B_teacher)
    yte_raw = act(s_te).detach().cpu().numpy()
    yte = (yte_raw - float(mean_y)) / float(std_y)

    ovH = float(mps.feature_overlap_corr_invariant(Hte_hat, Hte_true_for_overlap))
    corr_s = float(mps.corr_second_layer_scalar(ste_hat, ste_true))

    Bspec = mps.spectrum_metrics_B(Bhat_cpu, B_teacher)
    eig_err_B = float(Bspec["eig_err_B"])
    eig_corr_B = float(Bspec["eig_corr_B"])
    c_opt_B = float(Bspec["c_opt_B"])

    s_test = _raw_pred_from_input(X_or_Z_test, Ahat, Bhat_cpu, model, device).detach().cpu().numpy()
    yhat = s_test if is_id else estimators.predict_polynomial_link(s_test, coeffs, mu_sig=mu_sig)

    mse = float(np.mean((yhat - yte) ** 2))
    baseline = float(np.mean(yte ** 2))
    nmse = mse / (baseline + 1e-12)

    metrics = {
        "alpha": float(alpha),
        "n": int(n),
        "d": int(d),
        "p": int(p),
        "seed": int(seed),
        "model": str(model),
        "beta": float(beta),
        "A_mode_teacher": str(A_mode_teacher),
        "B_mode": str(B_mode),
        "gamma": float(gamma),
        "g_name": str(g_name),
        "batch_size": int(batch_size),
        "n_iter_C_max": int(n_iter_C_max),
        "n_iter_C_effective": int(info.get("n_iter_effective", n_iter_C_max)),
        "stop_delta": info.get("stop_delta", None),
        "oversamp_C": int(oversamp_C),
        "mean_y": float(mean_y),
        "std_y": float(std_y),
        "nmse": float(nmse),
        "mse": float(mse),
        "baseline": float(baseline),
        "ovA": None if ovA is None else float(ovA),
        "ovH": float(ovH),
        "corr_s": float(corr_s),
        "eig_err_B": float(eig_err_B),
        "eig_corr_B": float(eig_corr_B),
        "c_opt_B": float(c_opt_B),
        "wall_seconds": float(time.time() - t0),
        "device": str(device),
    }
    return metrics, Q_full


def _read_task_from_jsonl(taskfile: Path, task_id: int) -> dict:
    with taskfile.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == task_id:
                return json.loads(line)
    raise IndexError(f"task_id {task_id} out of range for {taskfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default=None)
    ap.add_argument("--taskfile", type=str, default=None)
    ap.add_argument("--task_id", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    if args.task_json is not None:
        task = json.loads(args.task_json)
    elif args.taskfile is not None and args.task_id is not None:
        task = _read_task_from_jsonl(Path(args.taskfile), args.task_id)
    else:
        raise SystemExit("Provide --task_json or (--taskfile and --task_id)")

    exp_id = str(task.get("exp_id", "2L"))
    chunk_id = int(task.get("chunk_id", args.task_id if args.task_id is not None else 0))
    seed = int(task["seed"])

    d = int(task.get("d", 400))
    eps = float(task.get("eps", 0.5))
    p = int(round(d ** eps))

    alphas = sorted([float(a) for a in task["alphas"]])
    models = list(task.get("models", ["true"]))

    beta = float(task.get("beta", 1.0))
    n_test = int(task.get("n_test", 2000))
    batch_size = int(task.get("batch_size", 2048))

    A_mode_teacher = str(task.get("A_mode_teacher", teacher.DEFAULT_A_MODE))
    B_mode = str(task.get("B_mode", teacher.DEFAULT_B_MODE))
    gamma = float(task.get("gamma", teacher.DEFAULT_GAMMA))
    g_name = task.get("g_name", "id")

    n_iter_C_max = int(task.get("n_iter_C_max", 15))
    oversamp_C = int(task.get("oversamp_C", 10))
    T_min = int(task.get("T_min", 10))
    stop_tol = task.get("stop_tol", None)
    stop_tol = None if stop_tol is None else float(stop_tol)

    fit_degree = int(task.get("fit_degree", 5))
    fit_ridge = float(task.get("fit_ridge", 1e-6))

    device = _default_device(task.get("device", None))

    seed_everything(seed, deterministic=args.deterministic)

    rng_te = np.random.default_rng(12345)
    Xte = torch.tensor(rng_te.normal(size=(n_test, d)).astype(np.float32), device=device)

    outdir = Path(args.outdir)
    jobdir = outdir / exp_id / f"chunk={chunk_id:04d}" / f"seed={seed:04d}"
    jobdir.mkdir(parents=True, exist_ok=True)

    metrics_path = jobdir / "metrics.jsonl"
    done = _load_done_pairs(metrics_path)

    meta = {
        "task": task,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
    }
    (jobdir / "task_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    Q_state = {m: None for m in models}
    for m in models:
        ckpt = jobdir / f"Q_last_{m}.pt"
        if ckpt.exists():
            try:
                Q_state[m] = torch.load(ckpt, map_location=device)
            except Exception:
                Q_state[m] = None

    with metrics_path.open("a", encoding="utf-8") as f:
        for alpha in alphas:
            for model in models:
                if (float(alpha), str(model)) in done:
                    continue

                metrics, Q_full = _run_one_alpha_model(
                    d=d, p=p, alpha=float(alpha), seed=seed, model=str(model),
                    beta=beta, n_test=n_test, batch_size=batch_size,
                    A_mode_teacher=A_mode_teacher, B_mode=B_mode, gamma=gamma,
                    g_name=str(g_name),
                    fit_degree=fit_degree, fit_ridge=fit_ridge,
                    n_iter_C_max=n_iter_C_max, oversamp_C=oversamp_C,
                    T_min=T_min, stop_tol=stop_tol,
                    Q_init=Q_state[model],
                    device=device, Xte=Xte
                )

                if Q_full is not None:
                    Q_state[model] = Q_full.detach()
                    torch.save(Q_state[model], jobdir / f"Q_last_{model}.pt")

                f.write(json.dumps(metrics) + "\n")
                f.flush()

                print(
                    f"chunk={chunk_id:04d} seed={seed:04d} model={model} "
                    f"alpha={alpha:.4g} n={metrics['n']} nmse={metrics['nmse']:.4g} "
                    f"ovA={metrics['ovA']} corr_s={metrics['corr_s']:.4g} eigErrB={metrics['eig_err_B']:.4g} "
                    f"iters={metrics['n_iter_C_effective']} delta={metrics.get('stop_delta', None)}"
                )


if __name__ == "__main__":
    main()