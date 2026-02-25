import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import teacher
import estimators
import measures as mps

def run_seq_sweep_alpha_using_your_C(
    d=400, eps=0.5,
    alphas=(0.6,0.8,1.0,1.2,1.4,1.6),
    reps=3,
    beta=1.0,
    n_test=2000,
    batch_size=2048,
    n_cap=200000,
    A_mode_teacher="sym_orth_frob",
    n_iter_C=15,
    oversamp_C=10,
    path="_overlaps_sequential.png",
    seed0=0,
    device=None,
    g_name="id", 
    g_callable=None,
    fit_degree =5,
    fit_ridge=1e-6,
    B_mode='dense',
    gamma: float = 0.0,
    models=("true","gauss")
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(models, str):
        models = (models,)
    models = list(models)
    allowed = {"true", "gauss"}
    bad = [m for m in models if m not in allowed]
    if bad:
        raise ValueError(f"Unknown models {bad}. Allowed: {sorted(allowed)}")
    
    is_id = (g_callable is None) and (g_name is None or g_name == "id")

    p = int(round(d**eps))
    # p = max(1, min(p, d))

    d = int(d)
    n_test = int(n_test)
    batch_size = int(batch_size)
    n_cap = int(n_cap) if n_cap is not None else None

    # fixed test set
    rng_te = np.random.default_rng(12345)
    Xte_np = rng_te.normal(size=(n_test, d)).astype(np.float32)
    Xte = torch.tensor(Xte_np, device=device)

    metrics_final = {
        m: {"nmse_mean": [], "nmse_std": [],
            "ovA_mean": [], "ovA_std": [],
            "ovH_mean": [], "ovH_std": [],
            "ovH2_mean": [], "ovH2_std": []}
        for m in models
    }
    ns = []

    for a in alphas:
        n = int(round(d**a))
        n = min(n, int(n_cap))
        n = max(n, 1)
        ns.append(n)

        metrics_alpha = {m: {"nmse": [], "ovA": [], "ovH": [], "ovH2": []} for m in models}

        for r in range(reps):
            seed = seed0 + 1000*r + int(100*a)

            for model in models:

                # ---- compute mean/std of y on THIS dataset (pass 1) ----
                mean_y, std_y = teacher.compute_mean_std_y_stream(
                    d=d, p=p, n=n, batch_size=batch_size,
                    A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
                    g_name=g_name, g_callable=g_callable, input_mode=model, B_mode=B_mode, gamma=gamma
                )

                std_y = torch.clamp(std_y, min=1e-3)

                # ---- stream factory that replays normalized y (pass 2) ----
                def stream_fn_factory():
                    def stream_fn():
                        for X_or_Z, H, y_norm, A_teacher, B in teacher.stream_batches_teacher_y_normalized(
                            d=d, p=p, n=n, batch_size=batch_size,
                            A_mode=A_mode_teacher, beta=beta, seed=seed, device=device,
                            mean_y=mean_y, std_y=std_y, input_mode=model,
                            g_name=g_name, g_callable=g_callable,
                            B_mode=B_mode, gamma=gamma
                        ):
                            yield X_or_Z, y_norm
                    return stream_fn
            

                # ---- Step 1: Ahat from C ----
                Ahat = estimators.top_p_eigmats_of_C(
                    stream_fn_factory=stream_fn_factory,
                    d=d, n_total=n, p=p,
                    n_iter=n_iter_C,
                    oversamp=oversamp_C,
                    device=device, 
                    input_mode=model
                )  # (p,d,d)

                # ---- Step 2: Bhat from y * H2(hhat) ----
                if model == 'true':
                    Bhat_cpu = estimators.estimate_Bhat_from_stream(
                        stream_fn=stream_fn_factory(),
                        Ahat=Ahat, p=p, n_total=n,
                        device=device
                    )  # (p,p) cpu float64
                else:
                    # ---- Step 2 (gauss): estimate B from STUDENT features Hhat = Z @ Ahat_flat.T ----
                    Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)

                    def Hhat_stream_factory():
                        def Hhat_stream():
                            for Z, y_norm in stream_fn_factory()():  # Z = X_or_Z in gauss mode
                                Z = Z.to(device=device, dtype=torch.float32)
                                Hhat = Z @ Ahat_flat.T
                                yield Hhat, y_norm
                        return Hhat_stream

                    Bhat_cpu = estimators.estimate_Bhat_from_H_stream(
                        stream_fn=Hhat_stream_factory(), p=p, n_total=n, device=device
                    )  # (p,p) cpu float64 
            
                def raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device):
                    """
                    Returns s = \hat h^T \hat B \hat h - tr(\hat B)
                    computed in the SAME way as test-time.
                    """
                    if model == "true":
                        # X_or_Z is X
                        Hhat = estimators.compute_hhat_from_X_and_Ahat(X_or_Z, Ahat).to(device=device, dtype=torch.float32)
                    else:
                        # X_or_Z is Z, but we recompute Hhat exactly as at test time
                        Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
                        Hhat = X_or_Z.to(device=device, dtype=torch.float32) @ Ahat_flat.T  # here X_or_Z = Z

                    Bhat_t = Bhat_cpu.to(device=device, dtype=torch.float32)
                    trB = torch.trace(Bhat_t)
                    s = torch.einsum("bp,pq,bq->b", Hhat, Bhat_t, Hhat) - trB
                    return s  # torch (bs,)

                if is_id:
                    # --- calibration stream (same as old runner): compute s_hat on a few batches ---
                    s_list = []
                    max_batches = 2

                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s = raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
                        s_list.append(s.detach().cpu().numpy())
                        if j + 1 >= max_batches:
                            break

                    s_cal = np.concatenate(s_list, axis=0)
                    scale = 1.0 / (np.std(s_cal) + 1e-12)
                    Bhat_cpu = Bhat_cpu * scale


                else:
                    s_list, y_list = [], []
                    max_batches = 10

                    for j, (X_or_Z, y_norm) in enumerate(stream_fn_factory()()):
                        s = raw_pred_from_input(X_or_Z, Ahat, Bhat_cpu, model, device)
                        s_list.append(s.detach().cpu().numpy())
                        y_list.append(y_norm.detach().cpu().numpy())
                        if j + 1 >= max_batches:
                            break

                    s_fit = np.concatenate(s_list, axis=0)
                    y_fit = np.concatenate(y_list, axis=0)

                    coeffs, mu_sig = estimators.fit_polynomial_link(
                        s_fit, y_fit, degree=fit_degree, ridge=fit_ridge
                    )

                # ---- True y on test (rebuild SAME teacher from seed) ----
                gen = torch.Generator(device=device); gen.manual_seed(seed)
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

                # -------- overlap(A) --------
                if A_mode_teacher != "sym_orth_frob":
                    raise ValueError("overlap(A) plot requires A_mode_teacher='sym_orth_frob' to have A_teacher['A']")
                Atrue = A_teacher["A"].detach()          # (p,d,d)
                ovA = mps.subspace_overlap_frob(Ahat.detach(), Atrue)

                act = teacher.get_activation_fn(g_name=g_name, g_callable=g_callable)

                if model == "true":
                    Hte_true = teacher.compute_h_from_X_torch(Xte, A_teacher).to(device=device, dtype=torch.float32)
                    Hte_hat = estimators.compute_hhat_from_X_and_Ahat(Xte, Ahat).to(device=device, dtype=torch.float32)
                    Hte_true_for_overlap = Hte_true
                    ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
                    ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)
                else:
                    Ahat_flat  = teacher.flatten_A_sym_for_H2_feature(Ahat).to(device=device, dtype=torch.float32)
                    Atrue_flat = teacher.flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device=device, dtype=torch.float32)

                    gen_te = torch.Generator(device=device); gen_te.manual_seed(seed + 9999)
                    Zte = torch.randn(n_test, Ahat_flat.shape[1], generator=gen_te, device=device, dtype=torch.float32)

                    Hte_hat = Zte @ Ahat_flat.T
                    Hte_true_for_overlap = Zte @ Atrue_flat.T
                    Hte_true = Hte_true_for_overlap     # même objet, c’est la vérité “features” gauss
                    ste_true = teacher.compute_y_from_H_torch(Hte_true, B_teacher).to(device=device, dtype=torch.float32)
                    ste_hat = teacher.compute_y_from_H_torch(Hte_hat, B_teacher).to(device=device, dtype=torch.float32)

                s_te = teacher.compute_y_from_H_torch(Hte_true, B_teacher)
                yte_raw = act(s_te).detach().cpu().numpy()
                yte = (yte_raw - float(mean_y)) / float(std_y)

                ovH = mps.feature_overlap_corr_invariant(Hte_hat, Hte_true_for_overlap)

                ovH2 = mps.feature_overlap_corr_invariant(
                    ste_hat[:, None],
                    ste_true[:, None],
                )

                # ---- overlap on B (functional, teacher-agnostic) ----
                # move Bhat to torch on device
                # Bhat = Bhat_cpu.to(device=device, dtype=torch.float32)
                # # Compare Bhat and B_teacher by their predictions on Hte_hat
                # sim = mps.B_functional_similarity(Bhat.to(torch.float64), B_teacher.to(torch.float64), Xte=Hte_hat.to(torch.float64), device=device)
                # # store nmse (lower is better) and optionally cos/corr
                # ovH2 = sim["cos"]  # keep a single scalar for compatibility (cosine of predictions)
                # # also keep numeric metrics in case you want to inspect them
                # ovH2_nmse = sim["nmse"]
                # ovH2_corr = sim["corr"]

                if model == "true":
                    X_or_Z_test = Xte
                else:
                    X_or_Z_test = Zte
                s_test = raw_pred_from_input(X_or_Z_test, Ahat, Bhat_cpu, model, device).detach().cpu().numpy()
                if is_id:
                    yhat = s_test
                else:
                    yhat = estimators.predict_polynomial_link(s_test, coeffs, mu_sig=mu_sig)

                mse = float(np.mean((yhat - yte)**2))
                baseline = float(np.mean(yte**2))
                nmse = mse / (baseline + 1e-12)

                metrics_alpha[model]["nmse"].append(nmse)
                metrics_alpha[model]["ovA"].append(ovA)
                metrics_alpha[model]["ovH"].append(ovH)
                metrics_alpha[model]["ovH2"].append(ovH2)

        for m in models:
            arr_nmse = np.asarray(metrics_alpha[m]["nmse"], dtype=float)
            arr_ovA  = np.asarray(metrics_alpha[m]["ovA"], dtype=float)
            arr_ovH  = np.asarray(metrics_alpha[m]["ovH"], dtype=float)
            arr_ovH2  = np.asarray(metrics_alpha[m]["ovH2"], dtype=float)

            metrics_final[m]["nmse_mean"].append(float(arr_nmse.mean()))
            metrics_final[m]["nmse_std"].append(float(arr_nmse.std(ddof=1)) if len(arr_nmse) > 1 else 0.0)

            metrics_final[m]["ovA_mean"].append(float(arr_ovA.mean()))
            metrics_final[m]["ovA_std"].append(float(arr_ovA.std(ddof=1)) if len(arr_ovA) > 1 else 0.0)

            metrics_final[m]["ovH_mean"].append(float(arr_ovH.mean()))
            metrics_final[m]["ovH_std"].append(float(arr_ovH.std(ddof=1)) if len(arr_ovH) > 1 else 0.0)

            metrics_final[m]["ovH2_mean"].append(float(arr_ovH2.mean()))
            metrics_final[m]["ovH2_std"].append(float(arr_ovH2.std(ddof=1)) if len(arr_ovH2) > 1 else 0.0)


        print(f"alpha={a:.2f} n={n}")
        for m in models:
            print(
                f"  {m:5s} | NMSE={metrics_final[m]['nmse_mean'][-1]:.3g}±{metrics_final[m]['nmse_std'][-1]:.2g} "
                f"| ovA={metrics_final[m]['ovA_mean'][-1]:.3g}±{metrics_final[m]['ovA_std'][-1]:.2g} "
                f"| ovH={metrics_final[m]['ovH_mean'][-1]:.3g}±{metrics_final[m]['ovH_std'][-1]:.2g} "
                f"| ovH2={metrics_final[m]['ovH2_mean'][-1]:.3g}±{metrics_final[m]['ovH2_std'][-1]:.2g}"
            )

    title = f"d={d}, eps={eps} (p={p}), reps={reps}, batch={batch_size}"

    if len(models) == 1:

        mps.plot_with_errorbars(
            x=alphas,
            mean = metrics_final["true"]["nmse_mean"],
            std = list( np.array(metrics_final["true"]["nmse_std"])/math.sqrt(reps)),
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="MSE",
            title=title,
            path=path+"_nmse.png",
            logy=False,
        )

        mps.plot_with_errorbars(
            x=alphas,
            mean = metrics_final["true"]["ovA_mean"],
            std = list( np.array(metrics_final["true"]["ovA_std"])/math.sqrt(reps)),
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(A) in [0,1]",
            title=title,
            path=path+"_A.png",
            logy=False,
        )

        mps.plot_with_errorbars(
            x=alphas,
            mean = metrics_final["true"]["ovH_mean"],
            std = list( np.array(metrics_final["true"]["ovH_std"])/math.sqrt(reps)),
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(H) in [0,1]",
            title=title,
            path=path+"_H.png",
            logy=False,
        )

        mps.plot_with_errorbars(
            x=alphas,
            mean = metrics_final["true"]["ovH2_mean"],
            std = list(np.array(metrics_final["true"]["ovH2_std"])/math.sqrt(reps)),
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(H2) in [0,1]",
            title=title,
            path=path+"_H2.png",
            logy=False,
        )

    else:

        mps.plot_two_with_errorbars(
            x=alphas,
            curves={
                "true": (metrics_final["true"]["nmse_mean"], list( np.array(metrics_final["true"]["nmse_std"])/math.sqrt(reps))),
                "gauss": (metrics_final["gauss"]["nmse_mean"], list( np.array(metrics_final["gauss"]["nmse_std"])/math.sqrt(reps))),
            },
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="MSE",
            title=title,
            path=path+"_nmse.png",
            logy=False,
        )

        mps.plot_two_with_errorbars(
            x=alphas,
            curves={
                "true": (metrics_final["true"]["ovA_mean"], list( np.array(metrics_final["true"]["ovA_std"])/math.sqrt(reps))),
                "gauss": (metrics_final["gauss"]["ovA_mean"], list(np.array(metrics_final["gauss"]["ovA_std"])/math.sqrt(reps))),
            },
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(A) in [0,1]",
            title=title,
            path=path+"_A.png",
            logy=False,
        )

        mps.plot_two_with_errorbars(
            x=alphas,
            curves={
                "true": (metrics_final["true"]["ovH_mean"], list( np.array(metrics_final["true"]["ovH_std"])/math.sqrt(reps))),
                "gauss": (metrics_final["gauss"]["ovH_mean"], list(np.array(metrics_final["gauss"]["ovH_std"])/math.sqrt(reps))),
            },
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(H) in [0,1]",
            title=title,
            path=path+"_H.png",
            logy=False,
        )

        mps.plot_two_with_errorbars(
            x=alphas,
            curves={
                "true": (metrics_final["true"]["ovH2_mean"], list(np.array(metrics_final["true"]["ovH2_std"])/math.sqrt(reps))),
                "gauss": (metrics_final["gauss"]["ovH2_mean"], list(np.array(metrics_final["gauss"]["ovH2_std"])/math.sqrt(reps))),
            },
            xlabel=r"$\alpha$ in $n=d^\alpha$",
            ylabel="Subspace overlap(H2) in [0,1]",
            title=title,
            path=path+"_H2.png",
            logy=False,
        )

    # ---- Save everything in ONE npz ----
    save_dict = {
        "alphas": np.asarray(alphas, dtype=float),
        "ns": np.asarray(ns, dtype=int),
        "d": np.int64(d),
        "eps": np.float64(eps),
        "p": np.int64(p),
        "reps": np.int64(reps),
        "beta": np.float64(beta),
        "n_test": np.int64(n_test),
        "batch_size": np.int64(batch_size),
    }

    for m in models:
        for k in ["nmse_mean","nmse_std","ovA_mean","ovA_std","ovH_mean","ovH_std","ovH2_mean","ovH2_std"]:
            save_dict[f"{m}_{k}"] = np.asarray(metrics_final[m][k], dtype=float)

    mps.save_npz(path + ".npz", **save_dict)


    return {"alphas": list(alphas), "ns": ns, "mse": metrics_final["true"]["nmse_mean"]}