import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


import teacher
import estimators
import measures as mps

def run_seq_sweep_alpha_using_your_C_3layers(
    d=400, eps1=0.5, eps2=0.5,
    alphas=(0.6,0.8,1.0,1.2,1.4,1.6),
    reps=3,
    beta=1.0,
    n_test=2000,
    batch_size=2048,
    n_cap=200000,
    A1_mode_teacher="sym_orth_frob",
    A2_mode_teacher="sym_orth_frob",
    n_iter_C=5,
    oversamp_C=3,
    path="_overlaps_sequential.png",
    seed0=0,
    device=None
):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    p1 = int(round(d**eps1))
    p2 = int(round(d**eps2))

    # fixed test set
    rng_te = np.random.default_rng(12345)
    Xte_np = rng_te.normal(size=(n_test, d)).astype(np.float32)
    Xte = torch.tensor(Xte_np, device=device)

    mse_mean, base_mean, ns = [], [], []
    nmse_mean, nmse_std = [], []
    # nmse_mean_2, nmse_std_2 = [], []
    # nmse_mean_24, nmse_std_24 = [], []

    ovA1_mean, ovA1_std = [], []
    ovH1_mean, ovH1_std = [], []
    ovA2_mean, ovA2_std = [], []
    ovH2_mean, ovH2_std = [], []
    ovB_mean,  ovB_std  = [], []



    for a in alphas:
        n = int(round(d**a))
        n = min(n, int(n_cap))
        n = max(n, 1)
        ns.append(n)

        mses, bases = [], []
        nmses = []
        # nmses2, nmses24 = [], []
        ovA1s = []
        ovH1s = []
        ovA2s = []
        ovH2s = []
        ovBs = []


        for r in range(reps):
            seed = seed0 + 1000*r + int(100*a)

            # ---- compute mean/std of y on THIS dataset (pass 1) ----
            mean_y, std_y = teacher.compute_mean_std_y_stream_3layers(
                d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
                A1_mode=A1_mode_teacher, A2_mode=A2_mode_teacher,
                beta=beta, seed=seed, device=device
            )

            # ---- stream factory that replays normalized y (pass 2) ----
            def stream_fn_factory():
                def stream_fn():
                    for X, H1, H2, y_norm, A1_teacher, A2_teacher, B_teacher in \
                        teacher.stream_batches_teacher_3layers_y_normalized(
                            d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
                            A1_mode=A1_mode_teacher, A2_mode=A2_mode_teacher,
                            beta=beta, seed=seed, device=device,
                            mean_y=mean_y, std_y=std_y
                        ):
                        yield X, y_norm
                return stream_fn

            # ---- Step 1: Ahat from C ----
            ## Power iteration version:
            A1_hat = estimators.top_p_eigmats_of_C(
                stream_fn_factory=stream_fn_factory,
                d=d, n_total=n, p=p1,
                n_iter=n_iter_C,
                oversamp=oversamp_C,
                device=device
            )  # (p,d,d)

            # ---- Step 2: h1hat from A1hat ----
            def stream_h1hat_y():
                for X, H1, H2, y_norm, *_ in \
                    teacher.stream_batches_teacher_3layers_y_normalized(
                        d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
                        A1_mode=A1_mode_teacher, A2_mode=A2_mode_teacher,
                        beta=beta, seed=seed, device=device,
                        mean_y=mean_y, std_y=std_y
                    ):
                    H1_hat = estimators.compute_hhat_from_X_and_Ahat(X, A1_hat)
                    yield H1_hat, y_norm

            def stream_h1hat_y_factory():
                def stream_fn():
                    yield from stream_h1hat_y()
                return stream_fn

            
            # ---- Step 3: A2hat from h1hat ----
            A2_hat = estimators.top_p_eigmats_of_C(
                stream_fn_factory=stream_h1hat_y_factory,
                d=p1, n_total=n, p=p2,
                n_iter=n_iter_C,
                oversamp=oversamp_C,
                device=device
            )


            # ---- Step 4: h2hat from A1hat ----
            def stream_h2hat_y():
                for H1_hat, y_norm in stream_h1hat_y():
                    H2_hat = estimators.compute_hhat_from_X_and_Ahat(H1_hat, A2_hat)
                    yield H2_hat, y_norm

            # ---- Step 5: B2hat from h2hat ----
            Bhat_cpu = estimators.estimate_Bhat_from_features_stream(
                stream_fn=stream_h2hat_y,   # callable
                p=p2, n_total=n,
                device=device
            )

            def stream_for_yhat_calib():
                for j, (X, H1, H2, y_norm, *_ ) in enumerate(
                    teacher.stream_batches_teacher_3layers_y_normalized(
                        d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
                        A1_mode=A1_mode_teacher, A2_mode=A2_mode_teacher,
                        beta=beta, seed=seed, device=device,
                        mean_y=mean_y, std_y=std_y
                    )
                ):
                    yield X
                    if j >= 1:
                        break

            # compute yhat on small calib set
            yhat_list = []
            Bhat_tmp = Bhat_cpu.to(device=device, dtype=torch.float32)
            trB_tmp = torch.trace(Bhat_tmp)

            for Xc in stream_for_yhat_calib():
                h1c = estimators.compute_hhat_from_X_and_Ahat(Xc, A1_hat)
                h2c = estimators.compute_hhat_from_X_and_Ahat(h1c, A2_hat)
                yhat_c = (torch.einsum("bp,pq,bq->b", h2c, Bhat_tmp, h2c) - trB_tmp).detach().cpu().numpy()
                yhat_list.append(yhat_c)

            yhat_c = np.concatenate(yhat_list, axis=0)
            scale = 1.0 / (np.std(yhat_c) + 1e-12)
            Bhat_cpu = Bhat_cpu * scale

            # # --- Naive estimation ---
            # yhat2, yhat24 = estimators.naive_predict_2_and_24(
            #     stream_fn_factory=stream_fn_factory,
            #     d=d,
            #     n_total=n,
            #     Xte=Xte,
            #     n4_cap=200_000_000_000,   # IMPORTANT si n est énorme; ajuste selon GPU
            #     device=device
            # )

            # ---- True y on test (rebuild SAME teacher from seed) ----
            gen = torch.Generator(device=device); gen.manual_seed(seed)

            if A1_mode_teacher == "rank1_orth":
                A1_teacher = teacher.gen_A_rank1_orth_torch(d, p1, gen, device)
            else:
                A1_teacher = teacher.gen_A_sym_orth_frob_torch(d, p1, gen, device)

            if A2_mode_teacher == "rank1_orth":
                A2_teacher = teacher.gen_A_rank1_orth_torch(p1, p2, gen, device)
            else:
                A2_teacher = teacher.gen_A_sym_orth_frob_torch(p1, p2, gen, device)

            B_teacher = teacher.gen_B_symmetric_dense_torch(p2, gen, device, beta=beta)

            # -------- overlap(A1) --------
            if A1_mode_teacher != "sym_orth_frob":
                raise ValueError("overlap(A) plot requires A1_mode_teacher='sym_orth_frob' to have A_teacher['A']")
            ovA1 = mps.subspace_overlap_frob(A1_hat.detach(), A1_teacher["A"])

            # -------- overlap(h1) --------
            H1_te_true = teacher.compute_h_from_X_torch(Xte, A1_teacher)
            H1_te_hat  = estimators.compute_hhat_from_X_and_Ahat(Xte, A1_hat)
            ovH1 = mps.feature_overlap_corr_invariant(H1_te_hat, H1_te_true)

            # -------- overlap(A2) --------
            if A2_mode_teacher != "sym_orth_frob":
                ovA2 = float("nan")
            else:
                ovA2 = mps.subspace_overlap_frob(A2_hat.detach(), A2_teacher["A"].detach())

            # -------- overlap(h2) --------
            H2_te_true = teacher.compute_h2_from_H1_torch(H1_te_true, A2_teacher)
            H2_te_hat  = estimators.compute_hhat_from_X_and_Ahat(H1_te_hat, A2_hat)
            ovH2 = mps.feature_overlap_corr_invariant(H2_te_hat, H2_te_true)

            # -------- overlap(B) (functional, teacher-agnostic) --------
            Bhat = Bhat_cpu.to(device=device, dtype=torch.float32)
            sim = mps.B_functional_similarity(Bhat.to(torch.float64), B_teacher.to(torch.float64), Xte=H2_te_hat.to(torch.float64), device=device)
            ovB = sim["cos"]

            trB = torch.trace(Bhat)
            yhat = (torch.einsum("bp,pq,bq->b", H2_te_hat, Bhat, H2_te_hat) - trB).detach().cpu().numpy()

            ovA1s.append(ovA1)
            ovA2s.append(ovA2)
            ovH1s.append(ovH1)
            ovH2s.append(ovH2)
            ovBs.append(ovB)

            # true features on test
            H1te_true = teacher.compute_h_from_X_torch(Xte, A1_teacher)                 # (n_test,p1)
            H2te_true = teacher.compute_h2_from_H1_torch(H1te_true, A2_teacher)         # (n_test,p2)

            # true y on test, then normalize with (mean_y,std_y) computed on training stream
            yte_raw = teacher.compute_y_from_H_torch(H2te_true, B_teacher).detach().cpu().numpy()
            yte = (yte_raw - float(mean_y)) / float(std_y)

            mse = float(np.mean((yhat - yte)**2))
            baseline = float(np.mean(yte**2))
            nmse = mse / (baseline + 1e-12)
            nmses.append(nmse)

            # nmse2  = float(np.mean((yhat2  - yte)**2) / (baseline + 1e-12))
            # nmses2.append(nmse2)
            # nmse24 = float(np.mean((yhat24 - yte)**2) / (baseline + 1e-12))
            # nmses24.append(nmse24)


        nmse_mean.append(float(np.mean(nmses)))
        nmse_std.append(float(np.std(nmses, ddof=1)) if reps > 1 else 0.0)

        ovA1_mean.append(float(np.mean(ovA1s)))
        ovA1_std.append(float(np.std(ovA1s, ddof=1)) if reps > 1 else 0.0)

        ovH1_mean.append(float(np.mean(ovH1s)))
        ovH1_std.append(float(np.std(ovH1s, ddof=1)) if reps > 1 else 0.0)

        ovA2_mean.append(float(np.mean(ovA2s)))
        ovA2_std.append(float(np.std(ovA2s, ddof=1)) if reps > 1 else 0.0)

        ovH2_mean.append(float(np.mean(ovH2s)))
        ovH2_std.append(float(np.std(ovH2s, ddof=1)) if reps > 1 else 0.0)

        ovB_mean.append(float(np.mean(ovBs)))
        ovB_std.append(float(np.std(ovBs, ddof=1)) if reps > 1 else 0.0)

        # nmse_mean_2.append(float(np.mean(nmses2)))
        # nmse_std_2.append(float(np.std(nmses2, ddof=1)) if reps > 1 else 0.0)

        # nmse_mean_24.append(float(np.mean(nmses24)))
        # nmse_std_24.append(float(np.std(nmses24, ddof=1)) if reps > 1 else 0.0)

        print(
          f"alpha={a:.2f} n={n} | NMSE={nmse_mean[-1]:.3g}±{nmse_std[-1]:.2g} "
          f"| ovA1={ovA1_mean[-1]:.3g}±{ovA1_std[-1]:.2g} "
          f"| ovH1={ovH1_mean[-1]:.3g}±{ovH1_std[-1]:.2g} "
          f"| ovA2={ovA2_mean[-1]:.3g}±{ovA2_std[-1]:.2g} "
          f"| ovH2={ovH2_mean[-1]:.3g}±{ovH2_std[-1]:.2g} "
          f"| ovB={ovB_mean[-1]:.3g}±{ovB_std[-1]:.2g}"
        )

    title = f"d={d}, eps1={eps1} (p1={p1}), eps2={eps2} (p2={p2}), reps={reps}, batch={batch_size}"

    # mps.plot_with_errorbars(
    #     x=alphas,
    #     mean=nmse_mean,
    #     std=nmse_std,
    #     xlabel=r"$\alpha$ in $n=d^\alpha$",
    #     ylabel="Sequential NMSE",
    #     title=title,
    #     path="y"+path,
    #     mean_else_1=nmse_mean_2,
    #     std_else_1=nmse_std_2,
    #     mean_else_2=nmse_mean_24,
    #     std_else_2=nmse_std_24,
    #     logy=True
    # )

    mps.plot_with_errorbars(
        x=alphas,
        mean=nmse_mean,
        std=nmse_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Sequential NMSE",
        title=title,
        path=path+"y",
        logy=False
    )

    mps.plot_with_errorbars(
        x=alphas,
        mean=ovA1_mean,
        std=ovA1_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Subspace overlap(A1) in [0,1]",
        title=title,
        path=path+"A1",
        logy=False
    )

    mps.plot_with_errorbars(
        x=alphas, mean=ovH1_mean, std=ovH1_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Feature overlap(h1) in [0,1]",
        title=title,
        path=path+"h1",
        logy=False
    )

    mps.plot_with_errorbars(
        x=alphas,
        mean=ovA2_mean,
        std=ovA2_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Subspace overlap(A2) in [0,1]",
        title=title,
        path=path+"A2",
        logy=False
    )

    mps.plot_with_errorbars(
        x=alphas, mean=ovH2_mean, std=ovH2_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Feature overlap(h2) in [0,1]",
        title=title,
        path=path+"h2",
        logy=False
    )

    mps.plot_with_errorbars(
        x=alphas, mean=ovB_mean, std=ovB_std,
        xlabel=r"$\alpha$ in $n=d^\alpha$",
        ylabel="Aligned overlap(B) (Frob cosine)",
        title=title,
        path=path+"B",
        logy=False
    )

    results = []
    for i, a in enumerate(alphas):
        results.append({
            "alpha": float(a),
            "n": int(ns[i]),
            "nmse_mean": float(nmse_mean[i]),
            "nmse_std": float(nmse_std[i]),
            "ovA1_mean": float(ovA1_mean[i]),
            "ovA1_std": float(ovA1_std[i]),
            "ovH1_mean": float(ovH1_mean[i]),
            "ovH1_std": float(ovH1_std[i]),
            "ovA2_mean": float(ovA2_mean[i]),
            "ovA2_std": float(ovA2_std[i]),
            "ovH2_mean": float(ovH2_mean[i]),
            "ovH2_std": float(ovH2_std[i]),
            "ovB_mean": float(ovB_mean[i]),
            "ovB_std": float(ovB_std[i]),
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)

    mps.save_results_txt(results, path + ".txt",
                        header={"d":d, "eps1":eps1, "p1":p1, "eps2":eps2, "p2":p2,
                                "reps":reps, "batch":batch_size, "beta":beta,
                                "A1_mode_teacher":A1_mode_teacher, "A2_mode_teacher":A2_mode_teacher,
                                "n_iter_C":n_iter_C, "oversamp_C":oversamp_C, "seed0":seed0, "n_cap":n_cap})
    mps.save_results_csv(results, path + ".csv")
    mps.save_results_json(results, path + ".json")

    return {"alphas": list(alphas), "ns": ns, "mse": mse_mean, "baseline": base_mean}