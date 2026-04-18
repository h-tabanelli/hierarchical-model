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
import math

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

def _raw_pred_from_Hhat(Hhat, Bhat_cpu, device):
    Bhat_t = Bhat_cpu.to(device=device, dtype=torch.float32)
    trB = torch.trace(Bhat_t)
    s = torch.einsum("bp,pq,bq->b", Hhat, Bhat_t, Hhat) - trB
    return s

def _maybe_load_saved_Ahat(
    outdir_root: Path,
    source_exp_id: str | None,
    alpha: float,
    seed: int,
    model: str,
    device,
):
    """
    Search for a previously saved Ahat from a spectral_B run:
      results/<source_exp_id>/chunk=*/seed=XXXX/artifacts/alpha=.../model=.../head=spectral_B/estimates.pt
    """
    if source_exp_id is None:
        return None

    root = Path(outdir_root) / str(source_exp_id)
    if not root.exists():
        return None

    pattern = (
        f"chunk=*/seed={seed:04d}/artifacts/"
        f"alpha={alpha:.4f}/model={model}/head=spectral_B/estimates.pt"
    )
    hits = sorted(root.glob(pattern))
    if not hits:
        return None

    payload = torch.load(hits[0], map_location=device)
    Ahat = payload.get("Ahat", None)
    if Ahat is None:
        return None

    return Ahat.to(device=device, dtype=torch.float32)


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
    head_mode: str,
    n_krr_max: int,
    rbf_lambda: float,
    rbf_sigma_mult: float,
    rbf_standardize: bool,
    poly_lambda: float,
    m_rf: int,
    n_iter_C_max: int,
    oversamp_C: int,
    T_min: int,
    stop_tol: float | None,
    Q_init: torch.Tensor | None,
    device: torch.device,
    Xte: torch.Tensor,
    jobdir: Path,
    save_estimates: bool = False,
    outdir_root: Path,
    load_ahat_exp_id: str | None,
    layer1_mode: str = "hermite_spectral",
    rf_width: int = 8192,
    rf_activation: str = "relu",
    calibrate_output: bool,
) -> tuple[dict, torch.Tensor | None]:
    """Run a single (alpha, model) and return (metrics_dict, Q_full_for_warm_start)."""
    t0 = time.time()

    n = int(round(d ** float(alpha)))
    n = max(n, 1)

    Ahat = None
    Vhat = None
    rf_layer = None
    whiten_mu = None
    whiten_cov = None
    whiten_mat = None

    act = teacher.get_activation_fn(g_name=g_name, g_callable=None)
    is_id = (g_name is None) or (g_name == "id")

    head_mode = str(head_mode)
    if head_mode not in {"spectral_B", "latent_rbf", "input_rbf", "latent_poly2", "input_poly4_rf"}:
        raise ValueError(f"Unknown head_mode: {head_mode}")

    layer1_mode = str(layer1_mode)
    if layer1_mode not in {"hermite_spectral", "rf_spectral"}:
        raise ValueError(f"Unknown layer1_mode: {layer1_mode}")

    if layer1_mode == "rf_spectral" and model != "true":
        raise ValueError("rf_spectral is implemented only for model='true'.")

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

    needs_Ahat = (layer1_mode == "hermite_spectral")

    # ---- Step 1: Ahat (warm start + early stop on iterations), or load from saved artifacts ----
    loaded_Ahat = False

    if layer1_mode == "hermite_spectral":
        if head_mode == "latent_rbf" and load_ahat_exp_id is not None:
            Ahat_loaded = _maybe_load_saved_Ahat(
                outdir_root=outdir_root,
                source_exp_id=load_ahat_exp_id,
                alpha=float(alpha),
                seed=int(seed),
                model=str(model),
                device=device,
            )
            if Ahat_loaded is not None:
                Ahat = Ahat_loaded
                Q_full = None
                info = {"n_iter_effective": 0, "stop_delta": None, "loaded_Ahat": True}
                loaded_Ahat = True

        if not loaded_Ahat:
            if stop_tol is None:
                Ahat = estimators.top_p_eigmats_of_C(
                    stream_fn_factory=stream_fn_factory,
                    d=d, n_total=n, p=p,
                    n_iter=n_iter_C_max,
                    oversamp=oversamp_C,
                    device=device,
                    input_mode=model,
                    Q_init=Q_init,
                )
                Q_full = None
                info = {"n_iter_effective": n_iter_C_max, "stop_delta": None, "loaded_Ahat": False}
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
                    return_info=True,
                )
                info["loaded_Ahat"] = False

    else:
        rf_seed_eff = int(seed) + 314159

        if stop_tol is None:
            rf_layer, Vhat = estimators.fit_rf_spectral_layer1_from_stream(
                stream_fn_factory=stream_fn_factory,
                d=d,
                rf_width=rf_width,
                p_out=p,
                n_total=n,
                rf_activation=rf_activation,
                rf_seed=rf_seed_eff,
                n_iter=n_iter_C_max,
                oversamp=oversamp_C,
                device=device,
                Q_init=Q_init,
                T_min=T_min,
                stop_tol=None,
            )
            Q_full = None
            info = {"n_iter_effective": n_iter_C_max, "stop_delta": None, "loaded_Ahat": False}
        else:
            rf_layer, Vhat, Q_full, info = estimators.fit_rf_spectral_layer1_from_stream(
                stream_fn_factory=stream_fn_factory,
                d=d,
                rf_width=rf_width,
                p_out=p,
                n_total=n,
                rf_activation=rf_activation,
                rf_seed=rf_seed_eff,
                n_iter=n_iter_C_max,
                oversamp=oversamp_C,
                device=device,
                Q_init=Q_init,
                T_min=T_min,
                stop_tol=stop_tol,
                return_Q_full=True,
                return_info=True,
            )
            info["loaded_Ahat"] = False

    # ---- Step 2: fit the head ----
    Bhat_cpu = None
    coeffs = None
    mu_sig = None
    rbf_model = None
    primal_model = None
    rf_map = None
    affine_a = None
    affine_b = None
    
    n_batches = (n + batch_size - 1) // batch_size

    calib_skip_batches = 0
    calib_take_batches = 0

    if calibrate_output and layer1_mode == "rf_spectral":
        calib_skip_batches = min(20, max(0, n_batches // 5))
        calib_take_batches = min(100, max(1, n_batches - calib_skip_batches))

    if head_mode == "spectral_B":
        if needs_Ahat:
            if model == "true":
                Bhat_cpu = estimators.estimate_Bhat_from_stream(
                    stream_fn=stream_fn_factory(),
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
        else:
            # ---- build raw RF latent stream ----
            def Hhat_stream_raw():
                for X, y_norm in stream_fn_factory()():
                    X = X.to(device=device, dtype=torch.float32)
                    Hhat = estimators.compute_hhat_from_X_and_rf(
                        X, rf_layer=rf_layer, Vhat=Vhat, device=device
                    )
                    yield Hhat, y_norm

            # ---- estimate whitening on raw RF latent ----
            whiten_mu, whiten_cov, whiten_mat = estimators.estimate_whitening_from_H_stream(
                stream_fn=Hhat_stream_raw,
                p=p,
                device=device,
                eps=1e-6,
            )

            # ---- whitened latent stream ----
            def Hhat_stream():
                for Hraw, y_norm in Hhat_stream_raw():
                    Hwhite = estimators.apply_whitening_to_H(
                        Hraw, whiten_mu, whiten_mat, device=device
                    )
                    yield Hwhite, y_norm

            # ---- standard isotropic second layer on whitened latent ----
            Bhat_cpu = estimators.estimate_Bhat_from_H_stream(
                stream_fn=Hhat_stream, p=p, n_total=n, device=device
            )

        # ---- optional calibration / link ----
        if layer1_mode == "rf_spectral":
            if calibrate_output:
                s_list, y_list = [], []
                for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                    if j < calib_skip_batches:
                        continue
                    if j >= calib_skip_batches + calib_take_batches:
                        break

                    Hhat_tmp = estimators.compute_hhat_from_X_and_rf_whitened(
                        X_or_Z.to(device=device, dtype=torch.float32),
                        rf_layer=rf_layer,
                        Vhat=Vhat,
                        whiten_mu_cpu=whiten_mu,
                        whiten_mat_cpu=whiten_mat,
                        device=device,
                    )
                    s = _raw_pred_from_Hhat(Hhat_tmp, Bhat_cpu, device)

                    s_list.append(s.detach().cpu().numpy())
                    y_list.append(y_norm.detach().cpu().numpy())

                if len(s_list) > 0:
                    s_cal = np.concatenate(s_list, axis=0)
                    y_cal = np.concatenate(y_list, axis=0)
                    affine_a, affine_b = estimators.fit_affine_link(s_cal, y_cal)

        else:
            if is_id:
                # legacy variance rescaling: keep it
                s_list = []
                scale_take_batches = min(2, n_batches)
                for j, (X_or_Z, _) in enumerate(stream_fn_factory()()):
                    s = _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
                    s_list.append(s.detach().cpu().numpy())
                    if j + 1 >= scale_take_batches:
                        break
                if len(s_list) > 0:
                    s_cal = np.concatenate(s_list, axis=0)
                    scale = 1.0 / (np.std(s_cal) + 1e-12)
                    Bhat_cpu = Bhat_cpu * float(scale)

                # optional affine calibration on top
                if calibrate_output:
                    s_list, y_list = [], []
                    affine_take_batches = min(10, n_batches)
                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s = _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
                        s_list.append(s.detach().cpu().numpy())
                        y_list.append(y_norm.detach().cpu().numpy())
                        if j + 1 >= affine_take_batches:
                            break
                    if len(s_list) > 0:
                        s_fit = np.concatenate(s_list, axis=0)
                        y_fit = np.concatenate(y_list, axis=0)
                        affine_a, affine_b = estimators.fit_affine_link(s_fit, y_fit)

            else:
                if calibrate_output:
                    s_list, y_list = [], []
                    poly_take_batches = min(10, n_batches)
                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s = _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
                        s_list.append(s.detach().cpu().numpy())
                        y_list.append(y_norm.detach().cpu().numpy())
                        if j + 1 >= poly_take_batches:
                            break
                    if len(s_list) > 0:
                        s_fit = np.concatenate(s_list, axis=0)
                        y_fit = np.concatenate(y_list, axis=0)
                        coeffs, mu_sig = estimators.fit_polynomial_link(
                            s_fit, y_fit, degree=fit_degree, ridge=fit_ridge
                        )

    elif head_mode == "latent_poly2":
        if not needs_Ahat:
            raise ValueError(f"{head_mode} currently requires layer1_mode='hermite_spectral'.")
        if model == "true":
            def feature_batch_fn(X_batch):
                Hhat = estimators.compute_hhat_from_X_and_Ahat(
                    X_batch.to(device=device, dtype=torch.float32), Ahat
                ).to(device=device, dtype=torch.float32)
                return estimators.build_latent_poly2_features(Hhat)
        else:
            Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)

            def feature_batch_fn(Z_batch):
                Hhat = Z_batch.to(device=device, dtype=torch.float32) @ Ahat_flat.T
                return estimators.build_latent_poly2_features(Hhat)

        primal_model = estimators.fit_primal_ridge_from_stream(
            stream_fn_factory=stream_fn_factory,
            feature_batch_fn=feature_batch_fn,
            lam=poly_lambda,
            device=device,
        )

    elif head_mode == "input_poly4_rf":
        if not needs_Ahat:
            raise ValueError(f"{head_mode} currently requires layer1_mode='hermite_spectral'.")
        rf_map = estimators.init_input_poly4_rf_map(d=d, m_rf=m_rf, device=device, seed=seed + 123456)

        def feature_batch_fn(X_or_Z_batch):
            return estimators.apply_input_poly4_rf_map(
                X_or_Z_batch.to(device=device, dtype=torch.float32),
                rf_map,
                device=device,
            )

        primal_model = estimators.fit_primal_ridge_from_stream(
            stream_fn_factory=stream_fn_factory,
            feature_batch_fn=feature_batch_fn,
            lam=poly_lambda,
            device=device,
        )

    # ---- rebuild teacher (for overlaps + test y) ----
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if A_mode_teacher == "rank1_orth":
        A_teacher = teacher.gen_A_rank1_orth_torch(d, p, gen, device)
    else:
        A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, device)

    if B_mode == "dense" or (B_mode == "powerlaw_diag" and float(gamma) == 0.0):
        B_teacher = teacher.gen_B_symmetric_dense_torch(p, gen, device, beta=beta)
    elif B_mode == "powerlaw_diag":
        B_teacher = teacher.gen_B_powerlaw_diag_torch(
            p, gen, device, beta=beta, gamma=gamma, rademacher=True
        )
    else:
        raise ValueError(f"Unknown B_mode: {B_mode}")

    ovA = None
    if needs_Ahat and A_mode_teacher == "sym_orth_frob":
        Atrue = A_teacher["A"].detach()
        ovA = float(mps.subspace_overlap_frob(Ahat.detach(), Atrue))

    if model == "true":
        Hte_true = teacher.compute_h_from_X_torch(Xte, A_teacher).to(device=device, dtype=torch.float32)
        # Hte_hat = estimators.compute_hhat_from_X_and_Ahat(Xte, Ahat).to(device=device, dtype=torch.float32)
        # Hte_true_for_overlap = Hte_true
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
        # ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        X_or_Z_test = Xte
        if needs_Ahat:
            Hte_hat = estimators.compute_hhat_from_X_and_Ahat(Xte, Ahat).to(device=device, dtype=torch.float32)
            Hte_true_for_overlap = Hte_true
            ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        else:
            Hte_hat = estimators.compute_hhat_from_X_and_rf_whitened(
                Xte,
                rf_layer=rf_layer,
                Vhat=Vhat,
                whiten_mu_cpu=whiten_mu,
                whiten_mat_cpu=whiten_mat,
                device=device,
            ).to(device=device, dtype=torch.float32)

            Hte_true_for_overlap = Hte_true
            ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
    else:
        # Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
        Atrue_flat = teacher.flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device=device, dtype=torch.float32)
        gen_te = torch.Generator(device=device)
        gen_te.manual_seed(seed + 9999)
        Zte = torch.randn(n_test, Ahat_flat.shape[1], generator=gen_te, device=device, dtype=torch.float32)
        # Hte_hat = Zte @ Ahat_flat.T
        # Hte_true_for_overlap = Zte @ Atrue_flat.T
        Hte_true = Zte @ Atrue_flat.T
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
        # ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        X_or_Z_test = Zte
        if needs_Ahat:
            Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
            Hte_hat = Zte @ Ahat_flat.T
            Hte_true_for_overlap = Hte_true
            ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        else:
            Hte_hat = None
            Hte_true_for_overlap = None
            ste_hat = None

    # ---- Optional: save estimates for posthoc analysis ----
    if save_estimates:
        artdir = jobdir / "artifacts" / f"alpha={alpha:.4f}" / f"model={model}" / f"head={head_mode}"
        artdir.mkdir(parents=True, exist_ok=True)

        payload = {
            "alpha": float(alpha),
            "d": int(d),
            "p": int(p),
            "seed": int(seed),
            "model": str(model),
            "head_mode": str(head_mode),
            "layer1_mode": str(layer1_mode),

            "Ahat": None if Ahat is None else Ahat.detach().to(device="cpu", dtype=torch.float16),

            "Vhat": None if Vhat is None else Vhat.detach().to(device="cpu", dtype=torch.float16),
            "Wrf": None if rf_layer is None else rf_layer["W"].detach().to(device="cpu", dtype=torch.float16),
            "rf_activation": None if rf_layer is None else str(rf_layer["rf_activation"]),

            "Bhat": None if Bhat_cpu is None else Bhat_cpu.detach().to(device="cpu", dtype=torch.float16),

            "whiten_mu": None if whiten_mu is None else whiten_mu.detach().to(device="cpu", dtype=torch.float32),
            "whiten_mat": None if whiten_mat is None else whiten_mat.detach().to(device="cpu", dtype=torch.float32),
            "whiten_cov": None if whiten_cov is None else whiten_cov.detach().to(device="cpu", dtype=torch.float32),

            "affine_a": affine_a,
            "affine_b": affine_b,

            "poly_coeffs": None if coeffs is None else np.asarray(coeffs, dtype=np.float64),
            "poly_mu_sig": None if mu_sig is None else (float(mu_sig[0]), float(mu_sig[1])),
        }
        torch.save(payload, artdir / "estimates.pt")

    s_te = teacher.compute_y_from_H_torch(Hte_true, B_teacher)
    yte_raw = act(s_te).detach().cpu().numpy()
    yte = (yte_raw - float(mean_y)) / float(std_y)

    if Hte_hat is not None and Hte_true_for_overlap is not None and ste_hat is not None:
        ovH = float(mps.feature_overlap_corr_invariant(Hte_hat, Hte_true_for_overlap))
        corr_s = float(mps.corr_second_layer_scalar(ste_hat, ste_true))
    else:
        ovH = None
        corr_s = None

    if head_mode == "spectral_B":
        Bspec = mps.spectrum_metrics_B(Bhat_cpu, B_teacher)
        eig_err_B = float(Bspec["eig_err_B"])
        eig_corr_B = float(Bspec["eig_corr_B"])
        c_opt_B = float(Bspec["c_opt_B"])

        if needs_Ahat:
            s_test = _raw_pred_from_input(X_or_Z_test, Ahat, Bhat_cpu, model, device).detach().cpu().numpy()
        else:
            s_test = _raw_pred_from_Hhat(Hte_hat, Bhat_cpu, device).detach().cpu().numpy()

        if layer1_mode == "rf_spectral":
            if calibrate_output and affine_a is not None and affine_b is not None:
                yhat = estimators.predict_affine_link(s_test, affine_a, affine_b)
            else:
                yhat = s_test
        else:
            if is_id:
                if calibrate_output and affine_a is not None and affine_b is not None:
                    yhat = estimators.predict_affine_link(s_test, affine_a, affine_b)
                else:
                    yhat = s_test
            else:
                if calibrate_output and coeffs is not None:
                    yhat = estimators.predict_polynomial_link(s_test, coeffs, mu_sig=mu_sig)
                else:
                    yhat = s_test

    elif head_mode in {"latent_rbf", "input_rbf"}:
        if not needs_Ahat:
            raise ValueError(f"{head_mode} currently requires layer1_mode='hermite_spectral'.")
        representation = "h1" if head_mode == "latent_rbf" else "x"

        Phi_train, y_train = estimators.collect_representation_train_from_stream(
            stream_fn_factory=stream_fn_factory,
            representation=representation,
            n_keep=(None if int(n_krr_max) <= 0 else min(int(n), int(n_krr_max))),
            model=model,
            Ahat=Ahat,
            device=device,
        )

        Phi_test = estimators.build_test_representation(
            X_or_Z_test,
            representation=representation,
            model=model,
            Ahat=Ahat,
            device=device,
        )

        if rbf_standardize:
            Phi_train, Phi_test, _, _ = estimators.standardize_features_train_test(Phi_train, Phi_test)

        rbf_model = estimators.fit_rbf_krr(
            Phi_train,
            y_train,
            sigma=None,
            lam=rbf_lambda,
            sigma_mult=rbf_sigma_mult,
            device=device,
        )

        yhat = estimators.predict_rbf_krr(rbf_model, Phi_test, device=device).numpy()

        eig_err_B = None
        eig_corr_B = None
        c_opt_B = None

    elif head_mode == "latent_poly2":
        Phi_test = estimators.build_latent_poly2_features(Hte_hat)
        yhat = estimators.predict_primal_ridge(primal_model, Phi_test, device=device).numpy()
        eig_err_B = None
        eig_corr_B = None
        c_opt_B = None

    elif head_mode == "input_poly4_rf":
        Phi_test = estimators.apply_input_poly4_rf_map(X_or_Z_test, rf_map, device=device)
        yhat = estimators.predict_primal_ridge(primal_model, Phi_test, device=device).numpy()
        eig_err_B = None
        eig_corr_B = None
        c_opt_B = None

    mse = float(np.mean((yhat - yte) ** 2))
    baseline = float(np.mean(yte ** 2))
    nmse = mse / (baseline + 1e-12)

    pred_mean = float(np.mean(yhat))
    pred_std = float(np.std(yhat))
    test_mean = float(np.mean(yte))
    test_std = float(np.std(yte))
    var_ratio = float((pred_std ** 2) / (test_std ** 2 + 1e-12))

    a_opt_pred = float(np.dot(yhat, yte) / (np.dot(yhat, yhat) + 1e-12))
    yhat_scaled = a_opt_pred * yhat
    mse_scaled = float(np.mean((yhat_scaled - yte) ** 2))
    nmse_scaled = mse_scaled / (baseline + 1e-12)

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
        "head_mode": str(head_mode),
        "n_krr_max": int(n_krr_max),
        "rbf_lambda": float(rbf_lambda),
        "rbf_sigma_mult": float(rbf_sigma_mult),
        "rbf_standardize": bool(rbf_standardize),
        "rbf_sigma": None if rbf_model is None else float(rbf_model["sigma"]),
        "rbf_sigma_base": None if rbf_model is None else float(rbf_model["sigma_base"]),
        "n_train_krr": None if rbf_model is None else int(rbf_model["n_train_krr"]),
        "loaded_Ahat": bool(info.get("loaded_Ahat", False)),
        "load_ahat_exp_id": None if load_ahat_exp_id is None else str(load_ahat_exp_id),
        "poly_lambda": float(poly_lambda),
        "m_rf": int(m_rf),
        "n_train_primal": None if primal_model is None else int(primal_model["n_train"]),
        "feature_dim": None if primal_model is None else int(primal_model["feature_dim"]),
        "layer1_mode": str(layer1_mode),
        "rf_width": int(rf_width),
        "rf_activation": str(rf_activation),
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
        "eig_err_B": None if eig_err_B is None else float(eig_err_B),
        "eig_corr_B": None if eig_corr_B is None else float(eig_corr_B),
        "c_opt_B": None if c_opt_B is None else float(c_opt_B),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "var_ratio": var_ratio,
        "a_opt_pred": a_opt_pred,
        "mse_scaled": mse_scaled,
        "nmse_scaled": nmse_scaled,
        "affine_a": None if affine_a is None else float(affine_a),
        "affine_b": None if affine_b is None else float(affine_b),
        "calib_skip_batches": int(calib_skip_batches),
        "calib_take_batches": int(calib_take_batches),
        "calibrate_output": bool(calibrate_output),
        "poly_coeffs_present": bool(coeffs is not None),
        "poly_mu": None if mu_sig is None else float(mu_sig[0]),
        "poly_sig": None if mu_sig is None else float(mu_sig[1]),
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
    head_mode = str(task.get("head_mode", "spectral_B"))
    n_krr_max = int(task.get("n_krr_max", 4000))
    rbf_lambda = float(task.get("rbf_lambda", 1e-4))
    rbf_sigma_mult = float(task.get("rbf_sigma_mult", 1.0))
    rbf_standardize = bool(task.get("rbf_standardize", True))
    poly_lambda = float(task.get("poly_lambda", 1e-4))
    m_rf = int(task.get("m_rf", 1024))
    layer1_mode = str(task.get("layer1_mode", "hermite_spectral"))
    rf_width = int(task.get("rf_width", 8192))
    rf_activation = str(task.get("rf_activation", "relu"))
    calibrate_output = bool(task.get("calibrate_output", False))

    load_ahat_exp_id = task.get("load_ahat_exp_id", None)
    if load_ahat_exp_id in {"", "none", "null"}:
        load_ahat_exp_id = None

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
                    device=device, Xte=Xte,
                    jobdir=jobdir, save_estimates=task.get("save_estimates", False),
                    head_mode=head_mode,
                    n_krr_max=n_krr_max,
                    rbf_lambda=rbf_lambda,
                    rbf_sigma_mult=rbf_sigma_mult,
                    rbf_standardize=rbf_standardize,
                    outdir_root=outdir,
                    load_ahat_exp_id=load_ahat_exp_id,
                    poly_lambda=poly_lambda,
                    m_rf=m_rf,
                    layer1_mode=layer1_mode,
                    rf_width=rf_width,
                    rf_activation=rf_activation,
                )

                if Q_full is not None:
                    Q_state[model] = Q_full.detach()
                    torch.save(Q_state[model], jobdir / f"Q_last_{model}.pt")

                f.write(json.dumps(metrics) + "\n")
                f.flush()

                eig_err_B_str = "None" if metrics["eig_err_B"] is None else f"{metrics['eig_err_B']:.4g}"
                print(
                    f"chunk={chunk_id:04d} seed={seed:04d} model={model} "
                    f"head={metrics['head_mode']} layer1={metrics['layer1_mode']} "
                    f"alpha={alpha:.4g} n={metrics['n']} nmse={metrics['nmse']:.4g} "
                    f"ovA={metrics['ovA']} corr_s={metrics['corr_s']:.4g} eigErrB={eig_err_B_str} "
                    f"iters={metrics['n_iter_C_effective']} delta={metrics.get('stop_delta', None)}"
                )


if __name__ == "__main__":
    main()