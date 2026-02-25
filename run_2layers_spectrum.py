import numpy as np
import torch
from pathlib import Path

import teacher
import estimators
import measures as mps


def tau(y):
    return y / (1.0 + torch.abs(y))


@torch.no_grad()
def run_2layers_spectra_per_alpha(
    d, p, n, batch_size,
    A_mode, beta,
    seed, device,
    out_dir,
    do_tau=True,
    bins=120,
    do_sanity=True,
    n_test=2000,
    g_name=None,
    g_callable=None,
    gamma: float = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- mean/std for y (TRUE teacher) --------
    mean_y, std_y = teacher.compute_mean_std_y_stream(
        d=d, p=p, n=min(n, 20000), batch_size=batch_size,
        A_mode=A_mode, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=g_callable,
        gamma=gamma,
    )

    # -------- get A_teacher, B from a first batch (TRUE) --------
    it = teacher.stream_batches_teacher_y_normalized(
        d=d, p=p, n=min(n, batch_size), batch_size=batch_size,
        A_mode=A_mode, beta=beta, seed=seed, device=device,
        mean_y=mean_y, std_y=std_y,
        g_name=g_name, g_callable=g_callable,
        gamma=gamma,
    )
    X0, H0, y0, A_teacher, B = next(iter(it))

    # Aflat maps tilde-x gaussian -> h
    Aflat = teacher.flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device)
    m = Aflat.shape[1]

    # k_top choice (not stupid)
    k_top_L1 = min(m, max(5*p, 30))
    k_top_L2 = min(p, 30)

    # ========= STREAMS =========

    # TRUE tilde-x (from x)
    def stream_xtilde_true():
        def _stream():
            for X, H, y_norm, _A, _B in teacher.stream_batches_teacher_y_normalized(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode=A_mode, beta=beta, seed=seed, device=device,
                mean_y=mean_y, std_y=std_y,
                g_name=g_name, g_callable=g_callable,
                gamma=gamma,
            ):
                Z = teacher.flatten_H2_of_X_sym_hermite(X)      # (bs, m)
                yield Z, y_norm
        return _stream

    # GAUSS-EQ tilde-x directly Gaussian
    def stream_xtilde_gauss():
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 777)
        def _stream():
            seen = 0
            while seen < n:
                bs = min(batch_size, n-seen)
                Z = torch.randn(bs, m, generator=gen, device=device)
                H = Z @ Aflat.T                                  # (bs,p)
                # apply power-law gamma scaling to H coordinates if requested
                if gamma is not None and gamma > 0:
                    p_loc = H.shape[1]
                    idx = torch.arange(1, p_loc+1, device=H.device, dtype=H.dtype)
                    factor = idx.pow(-float(gamma))
                    H = H * factor[None, :]
                # compute scalar pre-activation and apply activation if provided
                s = teacher.compute_y_from_H_torch(H, B)      # (bs,)
                act = teacher.get_activation_fn(g_name=g_name, g_callable=g_callable)
                y = act(s)
                y_norm = (y - mean_y.to(device)) / std_y.to(device)
                yield Z, y_norm
                seen += bs
        return _stream

    # TRUE h
    def stream_h_true():
        def _stream():
            for X, H, y_norm, _A, _B in teacher.stream_batches_teacher_y_normalized(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode=A_mode, beta=beta, seed=seed, device=device,
                mean_y=mean_y, std_y=std_y,
                g_name=g_name, g_callable=g_callable,
                gamma=gamma,
            ):
                yield H, y_norm
        return _stream

    # GAUSS-EQ h directly Gaussian
    def stream_h_gauss():
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 999)
        def _stream():
            seen = 0
            while seen < n:
                bs = min(batch_size, n-seen)
                H = torch.randn(bs, p, generator=gen, device=device)
                # apply power-law gamma scaling to H coordinates if requested
                if gamma is not None and gamma > 0:
                    idx = torch.arange(1, p+1, device=H.device, dtype=H.dtype)
                    factor = idx.pow(-float(gamma))
                    H = H * factor[None, :]
                s = teacher.compute_y_from_H_torch(H, B)
                act = teacher.get_activation_fn(g_name=g_name, g_callable=g_callable)
                y = act(s)
                y_norm = (y - mean_y.to(device)) / std_y.to(device)
                yield H, y_norm
                seen += bs
        return _stream

    # wrapper: apply tau to y
    def wrap_tau(stream_factory):
        def _factory():
            def _stream():
                for Z, y in stream_factory()():
                    yield Z, tau(y)
            return _stream
        return _factory
    
    if do_sanity:
        # ---------------------------
        # Sanity check (TRUE, no tau)
        # ---------------------------
        # small fixed test set
        rng_te = np.random.default_rng(12345)
        Xte_np = rng_te.normal(size=(n_test, d)).astype(np.float32)
        Xte = torch.tensor(Xte_np, device=device)

        # 1) estimate Ahat from the SAME (X,y_norm) stream (no tau)
        def stream_fn_factory_Xy():
            def stream_fn():
                for X, H, y_norm, _A, _B in teacher.stream_batches_teacher_y_normalized(
                    d=d, p=p, n=n, batch_size=batch_size,
                    A_mode=A_mode, beta=beta, seed=seed, device=device,
                    mean_y=mean_y, std_y=std_y
                ):
                    yield X, y_norm
            return stream_fn

        Ahat = estimators.top_p_eigmats_of_C(
            stream_fn_factory=stream_fn_factory_Xy,
            d=d, n_total=n, p=p,
            n_iter=15,
            oversamp=10,
            device=device
        )  # (p,d,d)

        # 2) estimate Bhat (standard estimator)
        Bhat_cpu = estimators.estimate_Bhat_from_stream(
            stream_fn=stream_fn_factory_Xy(),  # IMPORTANT: pass stream_fn, not factory
            Ahat=Ahat, p=p, n_total=n,
            device=device
        )  # cpu float64 (p,p)

        # 3) rebuild TRUE teacher params to compute y_true on test
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        if A_mode == "rank1_orth":
            A_teacher = teacher.gen_A_rank1_orth_torch(d, p, gen, device)
        else:
            A_teacher = teacher.gen_A_sym_orth_frob_torch(d, p, gen, device)
        B_teacher = teacher.gen_B_symmetric_dense_torch(p, gen, device, beta=beta)

        Hte_true = teacher.compute_h_from_X_torch(Xte, A_teacher)
        yte_raw = teacher.compute_y_from_H_torch(Hte_true, B_teacher).detach().cpu().numpy()
        yte = (yte_raw - float(mean_y)) / float(std_y)

        # 4) predict y_hat on test (sequential estimator)
        yhat = estimators.predict_y_from_Bhat_and_Ahat(
            Xte, Ahat, Bhat_cpu, device=device
        ).detach().cpu().numpy()

        mse = float(np.mean((yhat - yte) ** 2))
        baseline = float(np.mean(yte ** 2))
        nmse = mse / (baseline + 1e-12)

        # optional: overlap(A) if you can
        ovA = None
        if A_mode == "sym_orth_frob":
            Atrue = A_teacher["A"].detach()  # (p,d,d)
            ovA = float(mps.subspace_overlap_frob(Ahat.detach(), Atrue))

        print(f"[SANITY] n={n} | NMSE(y)={nmse:.4g}" + (f" | ovA={ovA:.4g}" if ovA is not None else ""))

        # save these quick diagnostics (append in your npz)
        sanity = dict(nmse_y=nmse)
        if ovA is not None:
            sanity["ovA"] = ovA


    # ========= SPECTRA COMPUTE =========
    # Layer1 matrices live in dim m ; Layer2 in dim p
    # dense_spectrum_from_stream returns ALL eigenvalues (what you want)

    specs = {}

    # L1
    specs["L1_true"] = estimators.dense_spectrum_from_stream(stream_xtilde_true, dim=m, n_total=n, device=device)
    specs["L1_ge"]   = estimators.dense_spectrum_from_stream(stream_xtilde_gauss, dim=m, n_total=n, device=device)

    if do_tau:
        specs["L1_true_T"] = estimators.dense_spectrum_from_stream(wrap_tau(stream_xtilde_true), dim=m, n_total=n, device=device)
        specs["L1_ge_T"]   = estimators.dense_spectrum_from_stream(wrap_tau(stream_xtilde_gauss), dim=m, n_total=n, device=device)

    # L2
    specs["L2_true"] = estimators.dense_spectrum_from_stream(stream_h_true, dim=p, n_total=n, device=device)
    specs["L2_ge"]   = estimators.dense_spectrum_from_stream(stream_h_gauss, dim=p, n_total=n, device=device)

    if do_tau:
        specs["L2_true_T"] = estimators.dense_spectrum_from_stream(wrap_tau(stream_h_true), dim=p, n_total=n, device=device)
        specs["L2_ge_T"]   = estimators.dense_spectrum_from_stream(wrap_tau(stream_h_gauss), dim=p, n_total=n, device=device)

    # ========= NORMALIZE SPECS (extract arrays) =========
    specs_eigs = {}   # name -> 1D np.array of eigenvalues
    specs_meta = {}   # name -> dict of extra info (mean_y, etc.) if present

    for name, obj in specs.items():
        if isinstance(obj, dict) and "evals" in obj:
            specs_eigs[name] = np.asarray(obj["evals"]).reshape(-1)
            specs_meta[name] = {k: v for k, v in obj.items() if k != "evals"}
        else:
            specs_eigs[name] = np.asarray(obj).reshape(-1)

    # ========= SAVE EIGENVALUES =========
    np.savez_compressed(out_dir / "eigs_all.npz", **specs_eigs)
    # optionnel : sauver les meta aussi
    np.savez_compressed(out_dir / "eigs_meta.npz", **{k: np.array(v, dtype=object) for k, v in specs_meta.items()})

    for name, evals in specs_eigs.items():
        print(name, type(evals), evals.shape)

    # ========= PLOTS =========
    for name, evals in specs_eigs.items():
        k_top = k_top_L1 if name.startswith("L1") else k_top_L2
        mps.plot_full_spectrum_with_top_abs(
            evals,
            title=f"{name} spectrum | d={d}, p={p}, n={n}",
            path=str(out_dir / f"{name}.png"),
            k_top=k_top,
            bins=bins
        )

    return specs_eigs

