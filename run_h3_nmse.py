# runner_2layers_h3_nmse.py
import math
from pathlib import Path
import numpy as np
import torch

import teacher as th3
import estimators as ev
import measures as mps

@torch.no_grad()
def compute_mean_std_y_from_stream(stream_fn_factory, device=None):
    s1 = torch.tensor(0.0, dtype=torch.float64, device=device)
    s2 = torch.tensor(0.0, dtype=torch.float64, device=device)
    n_seen = 0
    for y in stream_fn_factory()():  # yields y batches
        y = y.to(device=device, dtype=torch.float64).reshape(-1)
        s1 += y.sum()
        s2 += (y * y).sum()
        n_seen += y.numel()
    mean = (s1 / max(n_seen, 1)).item()
    var  = (s2 / max(n_seen, 1) - (s1 / max(n_seen, 1))**2).item()
    std  = float(np.sqrt(max(var, 0.0))) + 1e-12
    return mean, std


@torch.no_grad()
def row_orthonormalize(A_rows: torch.Tensor):
    """
    A_rows: (p,m). Returns rows orthonormal spanning same row-subspace.
    """
    # Orthonormalize rows by QR on transpose: A^T = Q R -> rows of Q^T are orthonormal
    Q, _ = torch.linalg.qr(A_rows.T)     # Q: (m,p)
    return Q.T.contiguous()             # (p,m)


@torch.no_grad()
def subspace_overlap_rows(A_hat_rows, A_true_rows):
    """
    Both (p,m) row-orthonormal.
    Returns ||A_hat A_true^T||_F^2 / p  in [0,1]
    """
    G = A_hat_rows @ A_true_rows.T
    return float((G.pow(2).sum() / A_hat_rows.shape[0]).item())


@torch.no_grad()
def estimate_Bhat_from_Hhat_stream(stream_fn_factory, p, n_total, device=None):
    """
    stream_fn_factory()() yields (Hhat, y) with Hhat:(bs,p), y:(bs,)
    Uses same convention as your old 2layers code:
        Bhat = (sum y h h^T - (sum y) I) / (2 n_total)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Sum = torch.zeros((p, p), device=device, dtype=torch.float64)
    Sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)

    seen = 0
    for Hhat, y in stream_fn_factory()():
        Hhat = Hhat.to(device=device, dtype=torch.float64)
        y    = y.to(device=device, dtype=torch.float64).reshape(-1)
        bs = y.numel()
        seen += bs

        Sum   += torch.einsum("b,bi,bj->ij", y, Hhat, Hhat)
        Sum_y += y.sum()

    I = torch.eye(p, device=device, dtype=torch.float64)
    Bhat = (Sum - Sum_y * I) / (2.0 * float(n_total))
    Bhat = 0.5 * (Bhat + Bhat.T)
    return Bhat


@torch.no_grad()
def y_from_H_B(H, B):
    # H:(n,p), B:(p,p)
    return (H @ B * H).sum(dim=1) - torch.trace(B)


@torch.no_grad()
def run_seq_sweep_alpha_H3(
    d: int,
    eps: float,
    alphas,
    reps: int,
    beta: float,
    n_test: int,
    batch_size: int,
    n_cap: int,
    n_iter_M: int,
    oversamp_M: int,
    out_dir: str,
    seed0: int,
    device,
    input_mode: str = "hermite3",
    normalize_y: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = int(round(d ** eps))
    m = d * (d + 1) * (d + 2) // 6

    rows = []

    for a in alphas:
        nmse_list = []
        ovA_list  = []
        ovH_list  = []
        ovB_list  = []

        n = int(round(d ** a))
        n = min(n, n_cap)

        for r in range(reps):
            seed = seed0 + 1000 * r + int(10_000 * a)

            # ---- get params once (A_true, B_true, idx,w) ----
            it = th3.stream_batches_2layers_H3(
                d=d, p=p, n=n, batch_size=batch_size,
                beta=beta,
                seed=seed,
                device=device,
                normalize_y=True,
                return_params=True
            )

            # IMPORTANT: generator is it()() in your convention
            X0, Z0, H0, y0, params = next(it())
            m = int(Z0.shape[1])
            A_true = params["A"]              # (p,m)
            assert A_true.shape[1] == m, f"A_true has m={A_true.shape[1]} but Z has m={m}"
            #print(params)
            if "B" not in params:
                raise KeyError("params['B'] missing: teacher.stream must return B in params.")
            B_true = params["B"]              # (p,p)
            idx = params.get("idx", None)
            w   = params.get("w", None)

            # ---- compute mean/std of TRAIN y (so test uses train stats) ----
            def stream_y_only():
                def _s():
                    it2 = th3.stream_batches_2layers_H3(
                        d=d, p=p, n=n, batch_size=batch_size, 
                        beta=beta, seed=seed,
                        device=device, normalize_y=True,
                        return_params=True
                    )
                    for _X, _Z, _H, y, _params in it2():
                        yield y
                return _s

            mean_y, std_y = 0.0, 1.0
            if normalize_y:
                mean_y, std_y = compute_mean_std_y_from_stream(stream_y_only, device=device)

            # ---- stream factory (Z, y_norm) ----
            def stream_zy():
                def _s():
                    it3 = th3.stream_batches_2layers_H3(
                        d=d, p=p, n=n, batch_size=batch_size,
                        beta=beta, seed=seed,
                        device=device, normalize_y=True,
                        return_params=True
                    )
                    for _X, Z, _H, y, _params in it3():
                        y_use = y
                        if normalize_y:
                            y_use = (y_use - mean_y) / std_y
                        yield Z, y_use
                return _s

            # =========================
            # (1) Estimate A (row-subspace) from operator on Z
            # =========================
            A_hat_rows = ev.top_p_eigvecs_abs_of_C_vec(
                stream_fn_factory=stream_zy,
                m=m, n_total=n, p=p,
                n_iter=n_iter_M, oversamp=oversamp_M,
                device=device
            )  # (p,m), rows orthonormal basis

            assert A_hat_rows.shape == (p, m), f"A_hat_rows shape {A_hat_rows.shape} expected {(p,m)}"

            # Orthonormalize A_true rows for clean subspace overlap
            A_true_rows = row_orthonormalize(A_true.to(device=device, dtype=torch.float64))
            A_hat_rows64 = A_hat_rows.to(device=device, dtype=torch.float64)

            # =========================
            # (2) Estimate B from (Hhat, y_norm) with external rotation handled later
            # =========================
            def stream_hhat_y():
                def _s():
                    for Z, y_use in stream_zy()():
                        Hhat = (Z.to(device=device) @ A_hat_rows.to(device=device, dtype=Z.dtype).T)   # (bs,p)
                        yield Hhat, y_use
                return _s

            B_hat = estimate_Bhat_from_Hhat_stream(stream_hhat_y, p=p, n_total=n, device=device)  # (p,p) float64

            # Optional: rescale B_hat so that predictions have std ~ 1 on TRAIN (since y is normalized)
            def stream_yhat_train():
                def _s():
                    for Hhat, _y in stream_hhat_y()():
                        yhat = y_from_H_B(Hhat.to(device=device, dtype=torch.float64), B_hat)
                        yield yhat
                return _s

            if normalize_y:
                _mp, std_pred = compute_mean_std_y_from_stream(stream_yhat_train, device=device)
                B_hat = B_hat / (std_pred + 1e-12)

            # =========================
            # (3) Test set: build Zte, Htrue/Hhat, predict y using true B and estimated B
            # =========================
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 9999)

            Xte = torch.randn(n_test, d, generator=gen, device=device)

            if input_mode == "gauss":
                Zte = torch.randn(n_test, m, generator=gen, device=device)
            else:
                Zte = th3.flatten_H3_sym(Xte)

            A_hat_rows = A_hat_rows.to(device=device, dtype=Zte.dtype)
            assert Zte.shape[1] == A_hat_rows.shape[1], f"dim mismatch: Zte={Zte.shape} vs A_hat_rows={A_hat_rows.shape}"

            H_true = (Zte @ A_true.to(device=device).T)                 # (n_test,p)
            H_hat  = (Zte @ A_hat_rows.to(device=device).T)             # (n_test,p)

            y_true = y_from_H_B(H_true.to(torch.float64), B_true.to(device=device, dtype=torch.float64))
            y_hat  = y_from_H_B(H_hat.to(torch.float64),  B_hat)

            if normalize_y:
                y_true = (y_true - mean_y) / std_y
                y_hat  = (y_hat  - mean_y) / std_y

            nmse = float(((y_hat - y_true).pow(2).mean() / (y_true.pow(2).mean() + 1e-12)).item())

            # =========================
            # (4) Overlaps
            # =========================
            ovA = subspace_overlap_rows(A_hat_rows64, A_true_rows)

            # rotation-invariant feature overlap (like old code)
            ovH = float(mps.feature_overlap_corr_invariant(
                H_hat.to(device=device, dtype=torch.float64),
                H_true.to(device=device, dtype=torch.float64)
            ))

            # Compare B_hat and B_true by their functional predictions on H_hat (teacher-agnostic)
            sim = mps.B_functional_similarity(
                B_hat.to(device=device, dtype=torch.float64),
                B_true.to(device=device, dtype=torch.float64),
                Xte=H_hat.to(device=device, dtype=torch.float64),
                device=device
            )
            ovB = float(sim["cos"])  # cosine between predicted vectors

            nmse_list.append(nmse)
            ovA_list.append(ovA)
            ovH_list.append(ovH)
            ovB_list.append(ovB)

        row = dict(
            d=d, p=p, m=m, alpha=float(a), n=int(n),
            nmse_mean=float(np.mean(nmse_list)), nmse_std=float(np.std(nmse_list)),
            ovA_mean=float(np.mean(ovA_list)),   ovA_std=float(np.std(ovA_list)),
            ovH_mean=float(np.mean(ovH_list)),   ovH_std=float(np.std(ovH_list)),
            ovB_mean=float(np.mean(ovB_list)),   ovB_std=float(np.std(ovB_list)),
            reps=reps, input_mode=input_mode, beta=beta, seed0=seed0
        )
        rows.append(row)
        print(f"alpha={a:.2f} n={n} | NMSE={row['nmse_mean']:.3g}±{row['nmse_std']:.3g} | "
              f"ovA={row['ovA_mean']:.3g}±{row['ovA_std']:.3g} | "
              f"ovH={row['ovH_mean']:.3g}±{row['ovH_std']:.3g} | "
              f"ovB={row['ovB_mean']:.3g}±{row['ovB_std']:.3g}")

    # ========= SAVE (npz + csv) =========
    np.savez_compressed(out_dir / "summary.npz", rows=np.array(rows, dtype=object))

    import csv
    csv_path = out_dir / "summary.csv"
    fieldnames = list(rows[0].keys()) if len(rows) > 0 else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)

    # ========= QUICK PLOTS =========
    import matplotlib.pyplot as plt
    al = np.array([rr["alpha"] for rr in rows], dtype=float)

    def _plot_metric(y_mean_key, y_std_key, fname, ylabel):
        y = np.array([rr[y_mean_key] for rr in rows], dtype=float)
        e = np.array([rr[y_std_key] for rr in rows], dtype=float)
        plt.figure(figsize=(6,4))
        plt.errorbar(al, y, yerr=e, marker="o", linestyle="-", linewidth=1.0, markersize=3, capsize=2)
        plt.xlabel("alpha")
        plt.ylim(-0.02, 1.02)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs alpha | d={d}, p={p} | mode={input_mode}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    _plot_metric("nmse_mean", "nmse_std", "nmse_vs_alpha.png", "NMSE(y)")
    _plot_metric("ovA_mean",  "ovA_std",  "ovA_vs_alpha.png",  "overlap(A)")
    _plot_metric("ovH_mean",  "ovH_std",  "ovH_vs_alpha.png",  "overlap(H)")
    _plot_metric("ovB_mean",  "ovB_std",  "ovB_vs_alpha.png",  "overlap(B)")

    print("Saved:", csv_path)
    print("Saved plots: nmse_vs_alpha.png, ovA_vs_alpha.png, ovH_vs_alpha.png, ovB_vs_alpha.png")

    return {"out_dir": str(out_dir), "rows": rows}