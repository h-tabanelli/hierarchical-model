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

# Ensure repo root is on sys.path when run as a script.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import teacher
import estimators
import measures as mps


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
        if h <= 0:
            raise ValueError("alpha step must be > 0")
        out, k = [], 0
        while True:
            val = a + k * h
            if val > b + 1e-12:
                break
            out.append(round(val, 10))
            k += 1
        return out
    return [float(x) for x in s.split(",") if x.strip()]


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Core computation (single alpha × model)
# ---------------------------------------------------------------------------

def _raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model: str, device):
    if model == "true":
        Hhat = estimators.compute_hhat_from_X_and_Ahat(
            X_or_Z.to(device=device, dtype=torch.float32), Ahat
        ).to(device=device, dtype=torch.float32)
    else:
        Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
        Hhat = X_or_Z.to(device=device, dtype=torch.float32) @ Ahat_flat.T

    Bhat_t = Bhat_cpu.to(device=device, dtype=torch.float32)
    trB = torch.trace(Bhat_t)
    return torch.einsum("bp,pq,bq->b", Hhat, Bhat_t, Hhat) - trB


def _raw_pred_from_Hhat(Hhat, Bhat_cpu, device):
    Bhat_t = Bhat_cpu.to(device=device, dtype=torch.float32)
    trB = torch.trace(Bhat_t)
    return torch.einsum("bp,pq,bq->b", Hhat, Bhat_t, Hhat) - trB


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
    device: torch.device,
    Xte: torch.Tensor,
    layer1_mode: str = "hermite_spectral",
    rf_width: int = 8192,
    rf_activation: str = "relu",
    rf2_width: int = 4096,
    rf2_activation: str = "relu_raw",
    rf2_affine_ridge: float = 1e-6,
    rf2_head_type: str = "vector_affine",
    calibrate_output: bool = False,
    rf_row_normalize_sphere: bool = False,
    rf2_row_normalize_sphere: bool = False,
    rf2_whiten_mode: str = "none",
) -> dict:
    t0 = time.time()

    n = max(int(round(d ** float(alpha))), 1)

    Ahat = None
    Vhat = None
    rf_layer = None
    rf2_layer = None
    rf2_affine_a = None
    rf2_affine_B = None
    ahat_rf2 = None
    h2_layer_metrics_vals = None
    whiten_mu = None
    whiten_cov = None
    whiten_mat = None
    whiten_std = None
    _whiten_mode_eff = "none"
    affine_a = None
    affine_b = None
    coeffs = None
    mu_sig = None
    rbf_model = None
    primal_model = None
    rf_map = None

    act = teacher.get_activation_fn(g_name=g_name, g_callable=None)
    is_id = (g_name is None) or (g_name == "id")

    if head_mode not in {"spectral_B", "latent_rbf", "input_rbf", "latent_poly2", "input_poly4_rf", "latent_rf_spectral"}:
        raise ValueError(f"Unknown head_mode: {head_mode}")
    if layer1_mode not in {"hermite_spectral", "rf_spectral"}:
        raise ValueError(f"Unknown layer1_mode: {layer1_mode}")
    if layer1_mode == "rf_spectral" and model != "true":
        raise ValueError("rf_spectral is implemented only for model='true'.")

    # ---- pass 1: mean/std ----
    mean_y, std_y = teacher.compute_mean_std_y_stream(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=None, input_mode=model,
        B_mode=B_mode, gamma=gamma,
    )
    std_y = torch.clamp(std_y, min=1e-3)

    def stream_fn_factory():
        def stream_fn():
            for X_or_Z, _, y_norm, _, _ in teacher.stream_batches_teacher_y_normalized(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
                mean_y=mean_y, std_y=std_y, input_mode=model,
                g_name=g_name, g_callable=None,
                B_mode=B_mode, gamma=gamma,
            ):
                yield X_or_Z, y_norm
        return stream_fn

    needs_Ahat = (layer1_mode == "hermite_spectral")

    # ---- Step 1: fit layer 1 ----
    if layer1_mode == "hermite_spectral":
        Ahat = estimators.top_p_eigmats_of_C(
            stream_fn_factory=stream_fn_factory,
            d=d, n_total=n, p=p,
            n_iter=n_iter_C_max,
            oversamp=oversamp_C,
            device=device,
            input_mode=model,
            Q_init=None,
        )
    else:
        rf_seed_eff = int(seed) + 314159
        rf_layer, Vhat = estimators.fit_rf_spectral_layer1_from_stream(
            stream_fn_factory=stream_fn_factory,
            d=d, rf_width=rf_width, p_out=p, n_total=n,
            rf_activation=rf_activation, rf_seed=rf_seed_eff,
            n_iter=n_iter_C_max, oversamp=oversamp_C,
            device=device, Q_init=None, T_min=0, stop_tol=None,
            normalize_rows=rf_row_normalize_sphere,
        )

    # ---- Step 2: build Hhat stream factory ----
    def _make_Hhat_stream_factory(whiten_mode_for_rf2: str):
        if needs_Ahat:
            if model == "true":
                def Hhat_stream():
                    for X_or_Z, y_norm in stream_fn_factory()():
                        yield estimators.compute_hhat_from_X_and_Ahat(
                            X_or_Z.to(device=device, dtype=torch.float32), Ahat
                        ).to(device=device, dtype=torch.float32), y_norm
                return lambda: Hhat_stream
            else:
                Ahat_flat_loc = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
                def Hhat_stream():
                    for X_or_Z, y_norm in stream_fn_factory()():
                        yield X_or_Z.to(device=device, dtype=torch.float32) @ Ahat_flat_loc.T, y_norm
                return lambda: Hhat_stream

        def Hhat_stream_raw():
            for X_or_Z, y_norm in stream_fn_factory()():
                yield estimators.compute_hhat_from_X_and_rf(
                    X_or_Z.to(device=device, dtype=torch.float32),
                    rf_layer=rf_layer, Vhat=Vhat, device=device,
                ), y_norm

        if whiten_mode_for_rf2 == "full":
            def Hhat_stream():
                for Hraw, y_norm in Hhat_stream_raw():
                    yield estimators.apply_whitening_to_H(Hraw, whiten_mu, whiten_mat, device=device), y_norm
            return lambda: Hhat_stream

        if whiten_mode_for_rf2 == "component":
            def Hhat_stream():
                for Hraw, y_norm in Hhat_stream_raw():
                    yield estimators.apply_whitening_componentwise_to_H(Hraw, whiten_mu, whiten_std, device=device), y_norm
            return lambda: Hhat_stream

        return lambda: Hhat_stream_raw

    n_batches = (n + batch_size - 1) // batch_size
    calib_skip_batches = 0
    calib_take_batches = 0
    if calibrate_output and layer1_mode == "rf_spectral":
        calib_skip_batches = min(20, max(0, n_batches // 5))
        calib_take_batches = min(100, max(1, n_batches - calib_skip_batches))

    # ---- Step 3: fit head ----
    Bhat_cpu = None

    if head_mode == "spectral_B":
        if needs_Ahat:
            if model == "true":
                Bhat_cpu = estimators.estimate_Bhat_from_stream(
                    stream_fn=stream_fn_factory(), Ahat=Ahat, p=p, n_total=n, device=device,
                )
            else:
                Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
                def Hhat_stream():
                    for Z, y_norm in stream_fn_factory()():
                        yield Z.to(device=device, dtype=torch.float32) @ Ahat_flat.T, y_norm
                Bhat_cpu = estimators.estimate_Bhat_from_H_stream(
                    stream_fn=Hhat_stream, p=p, n_total=n, device=device,
                )
        else:
            def Hhat_stream_raw_spec():
                for X, y_norm in stream_fn_factory()():
                    yield estimators.compute_hhat_from_X_and_rf(
                        X.to(device=device, dtype=torch.float32),
                        rf_layer=rf_layer, Vhat=Vhat, device=device,
                    ), y_norm

            whiten_mu, whiten_cov, whiten_mat = estimators.estimate_whitening_from_H_stream(
                stream_fn=Hhat_stream_raw_spec, p=p, device=device, eps=1e-6,
            )
            _whiten_mode_eff = "full"
            Hhat_stream_factory = _make_Hhat_stream_factory(whiten_mode_for_rf2="full")
            Bhat_cpu = estimators.estimate_Bhat_from_H_stream(
                stream_fn=Hhat_stream_factory(), p=p, n_total=n, device=device,
            )

        if layer1_mode == "rf_spectral" and calibrate_output:
            s_list, y_list = [], []
            for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                if j < calib_skip_batches:
                    continue
                if j >= calib_skip_batches + calib_take_batches:
                    break
                Hhat_tmp = estimators.compute_hhat_from_X_and_rf_whitened(
                    X_or_Z.to(device=device, dtype=torch.float32),
                    rf_layer=rf_layer, Vhat=Vhat,
                    whiten_mu_cpu=whiten_mu, whiten_mat_cpu=whiten_mat, device=device,
                )
                s = _raw_pred_from_Hhat(Hhat_tmp, Bhat_cpu, device)
                s_list.append(s.detach().cpu().numpy())
                y_list.append(y_norm.detach().cpu().numpy())
            if s_list:
                s_cal = np.concatenate(s_list)
                y_cal = np.concatenate(y_list)
                affine_a, affine_b = estimators.fit_affine_link(s_cal, y_cal)
        elif layer1_mode == "hermite_spectral":
            if is_id:
                s_list = []
                for j, (X_or_Z, _) in enumerate(stream_fn_factory()()):
                    s_list.append(_raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device).detach().cpu().numpy())
                    if j + 1 >= min(2, n_batches):
                        break
                if s_list:
                    s_cal = np.concatenate(s_list)
                    scale = 1.0 / (np.std(s_cal) + 1e-12)
                    Bhat_cpu = Bhat_cpu * float(scale)
                if calibrate_output:
                    s_list, y_list = [], []
                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s_list.append(_raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device).detach().cpu().numpy())
                        y_list.append(y_norm.detach().cpu().numpy())
                        if j + 1 >= min(10, n_batches):
                            break
                    if s_list:
                        affine_a, affine_b = estimators.fit_affine_link(
                            np.concatenate(s_list), np.concatenate(y_list)
                        )
            else:
                if calibrate_output:
                    s_list, y_list = [], []
                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s_list.append(_raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device).detach().cpu().numpy())
                        y_list.append(y_norm.detach().cpu().numpy())
                        if j + 1 >= min(10, n_batches):
                            break
                    if s_list:
                        coeffs, mu_sig = estimators.fit_polynomial_link(
                            np.concatenate(s_list), np.concatenate(y_list),
                            degree=fit_degree, ridge=fit_ridge,
                        )

    elif head_mode == "latent_rf_spectral":
        _whiten_mode_eff = rf2_whiten_mode if layer1_mode == "rf_spectral" else "none"

        def _raw_for_whiten():
            for X_or_Z, y_norm in stream_fn_factory()():
                yield estimators.compute_hhat_from_X_and_rf(
                    X_or_Z.to(device=device, dtype=torch.float32),
                    rf_layer=rf_layer, Vhat=Vhat, device=device,
                ), y_norm

        if _whiten_mode_eff == "full" and whiten_mu is None:
            whiten_mu, whiten_cov, whiten_mat = estimators.estimate_whitening_from_H_stream(
                stream_fn=_raw_for_whiten, p=p, device=device, eps=1e-6,
            )
        elif _whiten_mode_eff == "component" and whiten_mu is None:
            whiten_mu, whiten_std = estimators.estimate_whitening_componentwise_from_H_stream(
                stream_fn=_raw_for_whiten, p=p, device=device, eps=1e-6,
            )
        Hhat_stream_factory = _make_Hhat_stream_factory(whiten_mode_for_rf2=_whiten_mode_eff)

        if rf2_head_type == "simple":
            rf2_layer, ahat_rf2 = estimators.fit_rf_linear_head_from_H_stream(
                stream_fn_factory=Hhat_stream_factory, d_in=p,
                rf_width=rf2_width, n_total=n, rf_activation=rf2_activation,
                rf_seed=seed + 424242, device=device,
            )
        else:
            rf2_layer, rf2_affine_a, rf2_affine_B, ahat_rf2 = (
                estimators.fit_rf_vector_affine_removed_head_from_H_stream(
                    stream_fn_factory=Hhat_stream_factory, d_in=p,
                    rf_width=rf2_width, n_total=n, rf_activation=rf2_activation,
                    rf_seed=seed + 424242, device=device,
                    ridge=rf2_affine_ridge, normalize_rows=rf2_row_normalize_sphere,
                )
            )

        h2_chunks, y_chunks = [], []
        n_keep_h2 = None if int(n_krr_max) <= 0 else min(int(n), int(n_krr_max))
        kept_h2 = 0
        for H_batch, y_batch in Hhat_stream_factory()():
            if rf2_head_type == "simple":
                h2_batch = estimators.compute_h2hat_from_H_and_rf_linear_head(
                    H_batch.to(device=device, dtype=torch.float32),
                    rf_head=rf2_layer, ahat=ahat_rf2, device=device,
                ).reshape(-1, 1)
            else:
                h2_batch = estimators.compute_h2hat_from_H_and_rf_vector_affine_removed_linear_head(
                    H_batch.to(device=device, dtype=torch.float32),
                    rf_head=rf2_layer, a_aff=rf2_affine_a, B_aff=rf2_affine_B,
                    ahat=ahat_rf2, device=device,
                ).reshape(-1, 1)
            if n_keep_h2 is None:
                h2_chunks.append(h2_batch.detach().cpu())
                y_chunks.append(y_batch.detach().cpu())
            else:
                take = min(h2_batch.shape[0], n_keep_h2 - kept_h2)
                if take <= 0:
                    break
                h2_chunks.append(h2_batch[:take].detach().cpu())
                y_chunks.append(y_batch[:take].detach().cpu())
                kept_h2 += take
                if kept_h2 >= n_keep_h2:
                    break

        H2_train = torch.cat(h2_chunks, dim=0).to(torch.float32)
        y2_train = torch.cat(y_chunks, dim=0).to(torch.float32).reshape(-1)

        if is_id:
            if calibrate_output:
                affine_a, affine_b = estimators.fit_affine_link(H2_train[:, 0].numpy(), y2_train.numpy())
        else:
            rbf_model = estimators.fit_rbf_krr(
                H2_train, y2_train, sigma=None, lam=rbf_lambda,
                sigma_mult=rbf_sigma_mult, device=device,
            )

    elif head_mode == "latent_poly2":
        if not needs_Ahat:
            raise ValueError("latent_poly2 requires layer1_mode='hermite_spectral'.")
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
            stream_fn_factory=stream_fn_factory, feature_batch_fn=feature_batch_fn,
            lam=poly_lambda, device=device,
        )

    elif head_mode == "input_poly4_rf":
        if not needs_Ahat:
            raise ValueError("input_poly4_rf requires layer1_mode='hermite_spectral'.")
        rf_map = estimators.init_input_poly4_rf_map(d=d, m_rf=m_rf, device=device, seed=seed + 123456)
        def feature_batch_fn(X_or_Z_batch):
            return estimators.apply_input_poly4_rf_map(
                X_or_Z_batch.to(device=device, dtype=torch.float32), rf_map, device=device,
            )
        primal_model = estimators.fit_primal_ridge_from_stream(
            stream_fn_factory=stream_fn_factory, feature_batch_fn=feature_batch_fn,
            lam=poly_lambda, device=device,
        )

    elif head_mode in {"latent_rbf", "input_rbf"}:
        if not needs_Ahat:
            raise ValueError(f"{head_mode} requires layer1_mode='hermite_spectral'.")
        representation = "h1" if head_mode == "latent_rbf" else "x"
        Phi_train, y_train = estimators.collect_representation_train_from_stream(
            stream_fn_factory=stream_fn_factory, representation=representation,
            n_keep=(None if int(n_krr_max) <= 0 else min(int(n), int(n_krr_max))),
            model=model, Ahat=Ahat, device=device,
        )
        Phi_test = estimators.build_test_representation(
            Xte, representation=representation, model=model, Ahat=Ahat, device=device,
        )
        if rbf_standardize:
            Phi_train, Phi_test, _, _ = estimators.standardize_features_train_test(Phi_train, Phi_test)
        rbf_model = estimators.fit_rbf_krr(
            Phi_train, y_train, sigma=None, lam=rbf_lambda,
            sigma_mult=rbf_sigma_mult, device=device,
        )

    # ---- rebuild teacher for test evaluation ----
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if A_mode_teacher == "rank1_orth":
        A_teacher = teacher.gen_A_rank1_orth_torch(d, p, gen, device)
    else:
        A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, device)

    if B_mode == "dense" or (B_mode == "powerlaw_diag" and float(gamma) == 0.0):
        B_teacher = teacher.gen_B_symmetric_dense_torch(p, gen, device, beta=beta)
    elif B_mode == "powerlaw_diag":
        B_teacher = teacher.gen_B_powerlaw_diag_torch(p, gen, device, beta=beta, gamma=gamma, rademacher=True)
    else:
        raise ValueError(f"Unknown B_mode: {B_mode}")

    ovA = None
    if needs_Ahat and A_mode_teacher == "sym_orth_frob":
        Atrue = A_teacher["A"].detach()
        ovA = float(mps.subspace_overlap_frob(Ahat.detach(), Atrue))

    if model == "true":
        Hte_true = teacher.compute_h_from_X_torch(Xte, A_teacher).to(device=device, dtype=torch.float32)
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
        X_or_Z_test = Xte
        if needs_Ahat:
            Hte_hat = estimators.compute_hhat_from_X_and_Ahat(Xte, Ahat).to(device=device, dtype=torch.float32)
            Hte_true_for_overlap = Hte_true
            ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
        else:
            Hte_hat_raw = estimators.compute_hhat_from_X_and_rf(
                Xte, rf_layer=rf_layer, Vhat=Vhat, device=device,
            ).to(device=device, dtype=torch.float32)
            if _whiten_mode_eff == "full" and whiten_mu is not None:
                Hte_hat = estimators.apply_whitening_to_H(Hte_hat_raw, whiten_mu, whiten_mat, device=device)
            elif _whiten_mode_eff == "component" and whiten_mu is not None:
                Hte_hat = estimators.apply_whitening_componentwise_to_H(Hte_hat_raw, whiten_mu, whiten_std, device=device)
            else:
                Hte_hat = Hte_hat_raw
            Hte_hat = Hte_hat.to(device=device, dtype=torch.float32)
            Hte_true_for_overlap = Hte_true
            ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
    else:
        Atrue_flat = teacher.flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device=device, dtype=torch.float32)
        gen_te = torch.Generator(device=device)
        gen_te.manual_seed(seed + 9999)
        Zte = torch.randn(len(Xte), Ahat.reshape(p, -1).shape[1], generator=gen_te, device=device, dtype=torch.float32)
        Hte_true = Zte @ Atrue_flat.T
        ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
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

    ovH = None
    corr_s = None
    if Hte_hat is not None and Hte_true_for_overlap is not None and ste_hat is not None:
        ovH = float(mps.feature_overlap_corr_invariant(Hte_hat, Hte_true_for_overlap))
        corr_s = float(mps.corr_second_layer_scalar(ste_hat, ste_true))

    if head_mode == "latent_rf_spectral" and Hte_hat is not None:
        if rf2_head_type == "simple":
            h2_test = estimators.compute_h2hat_from_H_and_rf_linear_head(
                Hte_hat, rf_head=rf2_layer, ahat=ahat_rf2, device=device,
            )
        else:
            h2_test = estimators.compute_h2hat_from_H_and_rf_vector_affine_removed_linear_head(
                Hte_hat, rf_head=rf2_layer, a_aff=rf2_affine_a, B_aff=rf2_affine_B,
                ahat=ahat_rf2, device=device,
            )
        corr_s = float(mps.corr_second_layer_scalar(h2_test, ste_true))

    # ---- prediction on test set ----
    if head_mode == "spectral_B":
        Bspec = mps.spectrum_metrics_B(Bhat_cpu, B_teacher)
        eig_err_B = float(Bspec["eig_err_B"])
        eig_corr_B = float(Bspec["eig_corr_B"])
        c_opt_B = float(Bspec["c_opt_B"])

        s_test = (
            _raw_pred_from_input(X_or_Z_test, Ahat, Bhat_cpu, model, device)
            if needs_Ahat
            else _raw_pred_from_Hhat(Hte_hat, Bhat_cpu, device)
        ).detach().cpu().numpy()

        if layer1_mode == "rf_spectral":
            yhat = estimators.predict_affine_link(s_test, affine_a, affine_b) if (calibrate_output and affine_a is not None) else s_test
        elif is_id:
            yhat = estimators.predict_affine_link(s_test, affine_a, affine_b) if (calibrate_output and affine_a is not None) else s_test
        else:
            yhat = estimators.predict_polynomial_link(s_test, coeffs, mu_sig=mu_sig) if (calibrate_output and coeffs is not None) else s_test

    elif head_mode == "latent_rf_spectral":
        if rf2_head_type == "simple":
            h2_test = estimators.compute_h2hat_from_H_and_rf_linear_head(
                Hte_hat, rf_head=rf2_layer, ahat=ahat_rf2, device=device,
            ).detach().cpu().reshape(-1, 1)
        else:
            h2_test = estimators.compute_h2hat_from_H_and_rf_vector_affine_removed_linear_head(
                Hte_hat, rf_head=rf2_layer, a_aff=rf2_affine_a, B_aff=rf2_affine_B,
                ahat=ahat_rf2, device=device,
            ).detach().cpu().reshape(-1, 1)

        h2_layer_metrics_vals = mps.compute_h2_layer_metrics(
            h2_test[:, 0].numpy(), ste_true.detach().cpu().numpy(),
        )
        eig_err_B = eig_corr_B = c_opt_B = None

        if is_id:
            s_test = h2_test[:, 0].numpy()
            yhat = estimators.predict_affine_link(s_test, affine_a, affine_b) if (calibrate_output and affine_a is not None) else s_test
        else:
            yhat = estimators.predict_rbf_krr(rbf_model, h2_test, device=device).numpy()

    elif head_mode in {"latent_rbf", "input_rbf"}:
        yhat = estimators.predict_rbf_krr(rbf_model, Phi_test, device=device).numpy()
        eig_err_B = eig_corr_B = c_opt_B = None

    elif head_mode == "latent_poly2":
        Phi_test = estimators.build_latent_poly2_features(Hte_hat)
        yhat = estimators.predict_primal_ridge(primal_model, Phi_test, device=device).numpy()
        eig_err_B = eig_corr_B = c_opt_B = None

    elif head_mode == "input_poly4_rf":
        Phi_test = estimators.apply_input_poly4_rf_map(X_or_Z_test, rf_map, device=device)
        yhat = estimators.predict_primal_ridge(primal_model, Phi_test, device=device).numpy()
        eig_err_B = eig_corr_B = c_opt_B = None

    s_te = teacher.compute_y_from_H_torch(Hte_true, B_teacher)
    yte_raw = act(s_te).detach().cpu().numpy()
    yte = (yte_raw - float(mean_y)) / float(std_y)

    mse = float(np.mean((yhat - yte) ** 2))
    baseline = float(np.mean(yte ** 2))
    nmse = mse / (baseline + 1e-12)

    a_opt_pred = float(np.dot(yhat, yte) / (np.dot(yhat, yhat) + 1e-12))
    yhat_scaled = a_opt_pred * yhat
    nmse_scaled = float(np.mean((yhat_scaled - yte) ** 2)) / (baseline + 1e-12)

    return {
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
        "layer1_mode": str(layer1_mode),
        "n_iter_C_max": int(n_iter_C_max),
        "oversamp_C": int(oversamp_C),
        "rf_width": int(rf_width),
        "rf_activation": str(rf_activation),
        "rf2_width": int(rf2_width),
        "rf2_activation": str(rf2_activation),
        "rf2_affine_ridge": float(rf2_affine_ridge),
        "rf2_whiten_mode": str(rf2_whiten_mode),
        "batch_size": int(batch_size),
        "mean_y": float(mean_y),
        "std_y": float(std_y),
        "nmse": float(nmse),
        "mse": float(mse),
        "baseline": float(baseline),
        "nmse_scaled": float(nmse_scaled),
        "ovA": None if ovA is None else float(ovA),
        "ovH": None if ovH is None else float(ovH),
        "corr_s": None if corr_s is None else float(corr_s),
        "h2_pearson_r": None if h2_layer_metrics_vals is None else h2_layer_metrics_vals["pearson_r"],
        "h2_nmse_affine": None if h2_layer_metrics_vals is None else h2_layer_metrics_vals["nmse_affine"],
        "eig_err_B": None if eig_err_B is None else float(eig_err_B),
        "eig_corr_B": None if eig_corr_B is None else float(eig_corr_B),
        "c_opt_B": None if c_opt_B is None else float(c_opt_B),
        "rbf_sigma": None if rbf_model is None else float(rbf_model["sigma"]),
        "calibrate_output": bool(calibrate_output),
        "wall_seconds": float(time.time() - t0),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="2-layer hierarchical spectral estimator sweep over alpha = log_d(n).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--exp_id", type=str, default="2L",
                    help="Experiment identifier (used as output subdirectory name).")
    ap.add_argument("--d", type=int, default=100, help="Input dimension.")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="Hidden dimension exponent: p = round(d^eps).")
    ap.add_argument("--beta", type=float, default=1.0, help="Teacher B scale.")
    ap.add_argument("--alphas", type=str, required=True,
                    help='Alpha range: "start:stop:step" (e.g. "1.0:4.0:0.25") or CSV "1.0,2.0,3.0".')
    ap.add_argument("--seeds", type=str, default="0",
                    help='Seeds: range "0-4" or CSV "0,1,2".')
    ap.add_argument("--models", type=str, default="true,gauss",
                    help='Comma-separated list of input modes: "true" (Gaussian X) and/or "gauss" (Gaussian equivalent).')

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_test", type=int, default=5000)

    ap.add_argument("--A_mode_teacher", type=str, default="sym_orth_frob",
                    choices=["sym_orth_frob", "rank1_orth"])
    ap.add_argument("--B_mode", type=str, default="powerlaw_diag",
                    choices=["dense", "powerlaw_diag"])
    ap.add_argument("--gamma", type=float, default=0.25,
                    help="Power-law decay exponent for B eigenvalues (only for B_mode=powerlaw_diag).")
    ap.add_argument("--g_name", type=str, default="id",
                    help='Activation: "id" (identity / linear) or e.g. "relu", "erf".')

    ap.add_argument("--n_iter_C_max", type=int, default=15,
                    help="Number of randomized power-iteration passes for spectral estimation.")
    ap.add_argument("--oversamp_C", type=int, default=10,
                    help="Oversampling factor for randomized SVD.")

    ap.add_argument("--head_mode", type=str, default="spectral_B",
                    choices=["spectral_B", "latent_rbf", "input_rbf",
                             "latent_poly2", "input_poly4_rf", "latent_rf_spectral"])
    ap.add_argument("--layer1_mode", type=str, default="hermite_spectral",
                    choices=["hermite_spectral", "rf_spectral"])

    ap.add_argument("--rf_width", type=int, default=8192)
    ap.add_argument("--rf_activation", type=str, default="relu")
    ap.add_argument("--rf2_width", type=int, default=4096)
    ap.add_argument("--rf2_activation", type=str, default="relu_raw")
    ap.add_argument("--rf2_affine_ridge", type=float, default=1e-6)
    ap.add_argument("--rf2_head_type", type=str, default="vector_affine",
                    choices=["vector_affine", "simple"])
    ap.add_argument("--rf2_whiten_mode", type=str, default="full",
                    choices=["none", "component", "full"])
    ap.add_argument("--rf_row_normalize_sphere", action="store_true", default=False)
    ap.add_argument("--rf2_row_normalize_sphere", action="store_true", default=False)

    ap.add_argument("--n_krr_max", type=int, default=4000)
    ap.add_argument("--rbf_lambda", type=float, default=1e-4)
    ap.add_argument("--rbf_sigma_mult", type=float, default=1.0)
    ap.add_argument("--rbf_standardize", action="store_true", default=True)
    ap.add_argument("--no_rbf_standardize", dest="rbf_standardize", action="store_false")

    ap.add_argument("--poly_lambda", type=float, default=1e-4)
    ap.add_argument("--m_rf", type=int, default=1024)
    ap.add_argument("--fit_degree", type=int, default=5)
    ap.add_argument("--fit_ridge", type=float, default=1e-6)
    ap.add_argument("--calibrate_output", action="store_true", default=False)

    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--device", type=str, default=None,
                    help='PyTorch device string, e.g. "cuda:0" or "cpu". Auto-detected if omitted.')
    args = ap.parse_args()

    alphas = sorted(_parse_alphas(args.alphas))
    seeds = _parse_range(args.seeds)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    p = int(round(args.d ** args.eps))
    device = torch.device(args.device) if args.device else _default_device()

    print(f"d={args.d} eps={args.eps} p={p} device={device}")
    print(f"alphas ({len(alphas)}): {alphas[0]:.3g} .. {alphas[-1]:.3g}")
    print(f"seeds: {seeds}  models: {models}  head: {args.head_mode}  layer1: {args.layer1_mode}")

    outroot = Path(args.outdir)

    rng_te = np.random.default_rng(12345)
    Xte = torch.tensor(rng_te.normal(size=(args.n_test, args.d)).astype(np.float32), device=device)

    for seed in seeds:
        _seed_everything(seed)
        seeddir = outroot / args.exp_id / f"seed={seed:04d}"
        seeddir.mkdir(parents=True, exist_ok=True)
        metrics_path = seeddir / "metrics.jsonl"

        done: set[tuple[float, str]] = set()
        if metrics_path.exists():
            with metrics_path.open("r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        done.add((float(rec["alpha"]), str(rec["model"])))
                    except Exception:
                        pass

        with metrics_path.open("a") as f:
            for alpha in alphas:
                for model in models:
                    if (float(alpha), str(model)) in done:
                        print(f"  skip seed={seed} model={model} alpha={alpha:.4g} (already done)")
                        continue

                    metrics = _run_one_alpha_model(
                        d=args.d, p=p, alpha=alpha, seed=seed, model=model,
                        beta=args.beta, n_test=args.n_test, batch_size=args.batch_size,
                        A_mode_teacher=args.A_mode_teacher, B_mode=args.B_mode,
                        gamma=args.gamma, g_name=args.g_name,
                        fit_degree=args.fit_degree, fit_ridge=args.fit_ridge,
                        head_mode=args.head_mode, n_krr_max=args.n_krr_max,
                        rbf_lambda=args.rbf_lambda, rbf_sigma_mult=args.rbf_sigma_mult,
                        rbf_standardize=args.rbf_standardize,
                        poly_lambda=args.poly_lambda, m_rf=args.m_rf,
                        n_iter_C_max=args.n_iter_C_max, oversamp_C=args.oversamp_C,
                        device=device, Xte=Xte,
                        layer1_mode=args.layer1_mode,
                        rf_width=args.rf_width, rf_activation=args.rf_activation,
                        rf2_width=args.rf2_width, rf2_activation=args.rf2_activation,
                        rf2_affine_ridge=args.rf2_affine_ridge,
                        rf2_head_type=args.rf2_head_type,
                        calibrate_output=args.calibrate_output,
                        rf_row_normalize_sphere=args.rf_row_normalize_sphere,
                        rf2_row_normalize_sphere=args.rf2_row_normalize_sphere,
                        rf2_whiten_mode=args.rf2_whiten_mode,
                    )

                    f.write(json.dumps(metrics) + "\n")
                    f.flush()

                    eig_str = "N/A" if metrics["eig_err_B"] is None else f"{metrics['eig_err_B']:.4g}"
                    corr_str = "N/A" if metrics["corr_s"] is None else f"{metrics['corr_s']:.4g}"
                    print(
                        f"  seed={seed:04d} model={model} alpha={alpha:.4g} n={metrics['n']} "
                        f"nmse={metrics['nmse']:.4g} ovA={metrics['ovA']} "
                        f"corr_s={corr_str} eigErrB={eig_str} "
                        f"wall={metrics['wall_seconds']:.1f}s"
                    )


if __name__ == "__main__":
    main()
