# runner_2layers_h3_spectrum.py
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt



import teacher as th3
import estimators as ev


@torch.no_grad()
def run_h3_spectrum(
    d: int,
    eps: float,
    alpha: float,
    beta: float,
    batch_size: int,
    n_cap: int,
    out_dir: str,
    seed: int,
    device,
    input_mode: str = "hermite3",
    normalize_y: bool = True,
    full_eig_m_max: int = 6000,
    k_ritz: int = 200,
    n_iter_ritz: int = 12,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = int(round(d ** eps))
    m = d * (d + 1) * (d + 2) // 6
    n = int(round(d ** alpha))
    n = min(n, n_cap)

    # get params (A, idx, w)
    it = th3.stream_batches_2layers_H3(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode="orth",
        beta=beta,
        seed=seed,
        device=device,
        input_mode=input_mode,
        return_params=True
    )
    X0, Z0, H0, y0, params = next(it())
    A_true = params["A"]

    # stream factory for (Z,y_use)
    def stream_zy():
        def _s():
            for _X, Z, H, y, _params in th3.stream_batches_2layers_H3(
                d=d, p=p, n=n, batch_size=batch_size,
                A_mode="orth",
                beta=beta,
                seed=seed,
                device=device,
                input_mode=input_mode,
                return_params=True
            )():
                y_use = y
                if normalize_y:
                    y_use = y_use / (y_use.std() + 1e-12)
                yield Z, y_use
        return _s

    # --- spectrum of estimated operator Mhat ---
    if m <= full_eig_m_max:
        evals = ev.full_eigs_from_stream_if_small(stream_fn_factory=stream_zy, m=m, n_total=n, device=device)
        method = "full"
    else:
        evals = ev.ritz_eigs_of_C_vec(stream_fn_factory=stream_zy, m=m, n_total=n,
                                      k=k_ritz, n_iter=n_iter_ritz, oversamp=20, device=device)
        method = "ritz"

    # ---- SAVE EIGS (force 1D float array) ----
    evals = np.asarray(evals)
    if evals.ndim == 0:
        raise ValueError(f"[run_h3_spectrum] evals is scalar/0D: {evals} (method={method})")
    evals = evals.reshape(-1).astype(float)

    np.save(out_dir / "evals.npy", evals)

    # ---- METADATA ----
    meta = dict(
        d=d, p=p, m=m, alpha=float(alpha), n=int(n), beta=float(beta),
        input_mode=input_mode, normalize_y=bool(normalize_y),
        method=method, full_eig_m_max=int(full_eig_m_max),
        k_ritz=int(k_ritz), n_iter_ritz=int(n_iter_ritz), seed=int(seed)
    )
    np.savez_compressed(out_dir / "meta.npz", **meta)

    # ---- QUICK PLOT: histogram + vertical lines at top |eigs| ----
    def plot_full_spectrum_with_top_abs(evals, title, path, k_top=80, bins=120):
        evals = np.asarray(evals).reshape(-1)
        plt.figure(figsize=(7,4))
        plt.hist(evals, bins=bins, alpha=0.85)

        k = int(min(k_top, evals.size))
        if k > 0:
            top = evals[np.argsort(np.abs(evals))[-k:]]   # top |lambda|
            for v in top:
                plt.axvline(v, linewidth=0.35, linestyle="dashed")

        plt.title(title)
        plt.xlabel("eigenvalue  (dashed lines = top |eigs|)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # k_top: prends un truc simple (tu peux ajuster)
    k_top = p
    plot_full_spectrum_with_top_abs(
        evals,
        title=f"H3 spectrum ({method}) | d={d} p={p} m={m} n={n} alpha={alpha}",
        path=str(out_dir / "spectrum_hist.png"),
        k_top=k_top,
        bins=120
    )

    print(f"[H3 spectrum] d={d} p={p} m={m} alpha={alpha} n={n} -> {method} eigs saved to {out_dir}")
    print(f"[H3 spectrum] plot saved to {out_dir / 'spectrum_hist.png'}")

    return {"out_dir": str(out_dir), "meta": meta}

