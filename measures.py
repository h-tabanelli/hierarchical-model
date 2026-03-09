import math
import torch
import numpy as np
import os, json, time

### Overlaps definition ###

@torch.no_grad()
def subspace_overlap_frob(Ahat, Atrue):
    """
    Ahat: (p,d,d) orthonormal Frobenius
    Atrue:(p,d,d) orthonormal Frobenius (ou au moins span comparable)
    returns: overlap in [0,1] = (1/p) || Ahat_flat^T Atrue_flat ||_F^2
    """
    p,d,_ = Ahat.shape
    Ah = Ahat.reshape(p, d*d)      # (p, D)
    At = Atrue.reshape(p, d*d)     # (p, D)
    G = Ah @ At.T                  # (p,p)
    return (torch.linalg.norm(G, 'fro')**2 / p).item()

@torch.no_grad()
def feature_overlap_corr_invariant(Hhat, Htrue):
    """
    Invariant to column permutations/rotations:
      ov_h = (1/p) || Corr(Hhat, Htrue) ||_F^2  in [0,1]
    Inputs: Hhat,Htrue: (m,p) torch tensors
    """
    m, p = Hhat.shape
    Hh = (Hhat - Hhat.mean(0)) / (Hhat.std(0) + 1e-12)
    Ht = (Htrue - Htrue.mean(0)) / (Htrue.std(0) + 1e-12)
    C = (Hh.T @ Ht) / m            # (p,p), entries in [-1,1]
    return (torch.linalg.norm(C, 'fro')**2 / p).item()

@torch.no_grad()
def estimate_R_from_features(Hhat, Htrue):
    """
    Estimate orthogonal R that best aligns Htrue to Hhat in least squares:
      minimize ||Hhat - Htrue R^T||_F over orthogonal R
    returns R (p,p) torch tensor.
    """
    # Solve Procrustes: M = Hhat^T Htrue = U S V^T => R = U V^T
    M = Hhat.T @ Htrue
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return R

@torch.no_grad()
def B_overlap_aligned(Bhat, Btrue, R):
    """
    Compare Bhat to R Btrue R^T via Frobenius cosine.
    """
    Btarget = R @ Btrue @ R.T
    num = torch.sum(Bhat * Btarget)
    den = torch.linalg.norm(Bhat) * torch.linalg.norm(Btarget) + 1e-12
    return (num / den).item()


@torch.no_grad()
def B_functional_similarity(Bhat, Btrue, Xte=None, n_test=1000, device=None):
    """
    Functional comparison of Bhat and Btrue via their predictions on Xte.
    Returns a dict with:
      - nmse : normalized MSE between y_hat(Bhat) and y_true(Btrue)
      - corr : Pearson correlation between the predicted vectors
      - cos  : cosine between the flattened prediction vectors

    If Xte is None, samples Xte ~ N(0,I_d) with n_test rows; dimension d is inferred
    from Btrue shape (d = ???) — here Btrue is (p,p) so Xte should be features H (bs,p).
    Therefore Xte must be H (not raw x). If Xte is None, this function will generate
    random H ~ N(0,1) of shape (n_test, p).
    """
    # Bhat, Btrue: (p,p)
    if device is None:
        device = Bhat.device if hasattr(Bhat, 'device') else torch.device('cpu')

    p = Bhat.shape[0]

    if Xte is None:
        Hte = torch.randn(n_test, p, device=device)
    else:
        Hte = Xte.to(device=device)

    # compute predictions: y = h^T B h - tr(B)
    def preds(H, B):
        B = B.to(device=device, dtype=H.dtype)
        tr = torch.trace(B)
        y = torch.einsum("bp,pq,bq->b", H, B, H) - tr
        return y.detach().cpu().numpy()

    y_hat = preds(Hte, Bhat)
    y_true = preds(Hte, Btrue)

    # NMSE (normalized by var of y_true)
    mse = float(((y_hat - y_true)**2).mean())
    var = float(np.var(y_true)) + 1e-12
    nmse = mse / var

    # correlation
    if y_hat.std() < 1e-12 or y_true.std() < 1e-12:
        corr = float('nan')
    else:
        corr = float(np.corrcoef(y_hat, y_true)[0,1])

    # cosine between flattened prediction vectors
    num = float((y_hat @ y_true))
    den = (np.linalg.norm(y_hat) * np.linalg.norm(y_true) + 1e-12)
    cos = float(num / den)

    return {"nmse": nmse, "corr": corr, "cos": cos}


@torch.no_grad()
def B_conjugation_invariant_distance(Bhat, Btrue):
    """
    Conjugation-invariant distance between symmetric matrices Bhat and Btrue.
    Simple implementation: compare sorted eigenvalues (L2 relative).
    Returns relative L2 distance between sorted eigenvalues.
    """
    import numpy as _np
    bh = Bhat.detach().cpu().numpy()
    bt = Btrue.detach().cpu().numpy()
    eig_h = _np.linalg.eigvalsh(bh)
    eig_t = _np.linalg.eigvalsh(bt)
    # sort descending absolute (or ascending)
    eig_h = np.sort(eig_h)
    eig_t = np.sort(eig_t)
    # pad if different sizes (shouldn't happen)
    L = max(eig_h.size, eig_t.size)
    if eig_h.size < L:
        eig_h = np.pad(eig_h, (L-eig_h.size, 0), constant_values=0.0)
    if eig_t.size < L:
        eig_t = np.pad(eig_t, (L-eig_t.size, 0), constant_values=0.0)
    diff = eig_h - eig_t
    rel = np.linalg.norm(diff) / (np.linalg.norm(eig_t) + 1e-12)
    return float(rel)

# =========================
# B-metrics (rotation-invariant)
# =========================

import torch


@torch.no_grad()
def spectrum_metrics_B(B_hat: torch.Tensor, B_true: torch.Tensor) -> dict:
    """Rotation-invariant comparison of B matrices via eigenvalues, up to global scaling.

    Returns:
      - c_opt: best L2 scaling minimizing ||c*lam_hat - lam_true||_2
      - eig_err_B: relative L2 error on spectra after scaling
      - eig_corr_B: centered correlation between spectra
    """
    B_hat = (B_hat + B_hat.T) / 2
    B_true = (B_true + B_true.T) / 2

    lam_hat = torch.linalg.eigvalsh(B_hat.to(dtype=torch.float64, device="cpu"))
    lam_true = torch.linalg.eigvalsh(B_true.to(dtype=torch.float64, device="cpu"))

    den = (lam_hat @ lam_hat).item()
    c_opt = 0.0 if den < 1e-30 else float((lam_true @ lam_hat).item() / den)

    eig_err_B = float(
        torch.linalg.norm(c_opt * lam_hat - lam_true).item()
        / (torch.linalg.norm(lam_true).item() + 1e-12)
    )

    lam_hat_c = lam_hat - lam_hat.mean()
    lam_true_c = lam_true - lam_true.mean()
    eig_corr_B = float(
        (lam_hat_c @ lam_true_c).item()
        / ((torch.linalg.norm(lam_hat_c).item() + 1e-12) * (torch.linalg.norm(lam_true_c).item() + 1e-12))
    )

    return {"c_opt_B": float(c_opt), "eig_err_B": eig_err_B, "eig_corr_B": eig_corr_B}


@torch.no_grad()
def corr_second_layer_scalar(s_hat: torch.Tensor, s_true: torch.Tensor) -> float:
    """Correlation-style score for 2nd-layer scalar preactivation s(x).
    Expects shape (n,) or (n,1).
    """
    if s_hat.dim() == 1:
        s_hat = s_hat[:, None]
    if s_true.dim() == 1:
        s_true = s_true[:, None]
    return float(feature_overlap_corr_invariant(s_hat, s_true))


### Plot functions ###

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    print("Saved", path)

def plot_hist_flat(H, title, path, bins=80):
    """
    Histogramme des entries H_{mu,i} aplaties.
    """
    import matplotlib.pyplot as plt
    z = np.asarray(H).reshape(-1)
    plt.figure()
    plt.hist(z, bins=bins, density=True)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    savefig(path)

def plot_cov_diag(H, title, path):
    import matplotlib.pyplot as plt
    H = np.asarray(H)
    Hc = H - H.mean(axis=0, keepdims=True)
    cov = (Hc.T @ Hc) / Hc.shape[0]
    diag = np.diag(cov)
    plt.figure()
    plt.plot(diag, marker="o")
    plt.title(title)
    plt.xlabel("i")
    plt.ylabel("diag(cov)")
    savefig(path)

def plot_cov_spectrum(H, title, path):
    import matplotlib.pyplot as plt
    """
    Spectre de la covariance empirique de H.
    """
    H = np.asarray(H)
    Hc = H - H.mean(axis=0, keepdims=True)
    cov = (Hc.T @ Hc) / Hc.shape[0]
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    plt.figure()
    plt.plot(evals, marker="o")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("rank")
    plt.ylabel("eigenvalue (log)")
    savefig(path)

def save_npz(path, **arrays):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, **arrays)
    print("Saved", path)

# From curve to histogram 

def hist_from_density(grid, density, bins=80):
    # bins uniformes sur l’intervalle couvert par grid
    edges = np.linspace(grid.min(), grid.max(), bins+1)
    probs = np.zeros(bins)

    # intégrale par bin via trapz sur les points du bin
    for i in range(bins):
        a, b = edges[i], edges[i+1]
        mask = (grid >= a) & (grid < b)
        if mask.sum() >= 2:
            probs[i] = np.trapz(density[mask], grid[mask])
        else:
            probs[i] = 0.0

    # convertir en "hauteur d'histogramme" (densité) : prob / largeur
    widths = edges[1:] - edges[:-1]
    heights = probs / widths
    centers = 0.5*(edges[1:] + edges[:-1])
    return centers, heights, edges

def plot_density_and_hist(grid, density, title="", bins=80):
    import matplotlib.pyplot as plt

    centers, heights, edges = hist_from_density(grid, density, bins=bins)

    plt.figure(figsize=(7,4))
    # histogramme style RMT
    plt.bar(centers, heights, width=(edges[1]-edges[0]), alpha=0.4, align='center', edgecolor='k', linewidth=0.3)
    # courbe lisse par-dessus (optionnel)
    plt.plot(grid, density, linewidth=2)
    plt.title(title)
    plt.xlabel("eigenvalue")
    plt.ylabel("density")
    plt.tight_layout()
    plt.show()

def plot_full_spectrum(evals, title, path, bins=120, logy=False):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(evals, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("eigenvalue")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_full_spectrum_with_top(evals, title, path, k_top=40, bins=120):
    import matplotlib.pyplot as plt
    evals = np.asarray(evals)
    plt.figure(figsize=(7,4))
    plt.hist(evals, bins=bins, alpha=0.8)

    # indices des k plus grandes |lambda|
    idx = np.argsort(np.abs(evals))[-k_top:]
    top = np.sort(evals[idx])  # valeurs signées correspondantes, triées pour un plot propre

    for v in top:
        plt.axvline(v, linewidth=0.2, linestyle='dashed')

    plt.title(title)
    plt.xlabel("eigenvalue (vertical lines = top |eigs|)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_with_errorbars(x, mean, std, xlabel, ylabel, title, path, logy=False, marker='o', color='C0'):
    import matplotlib.pyplot as plt
    """
    Simple plot with mean and errorbars (std). Saves figure to `path`.
    `x`, `mean`, `std` can be lists or numpy arrays.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    std = np.asarray(std)

    plt.figure(figsize=(7,4))
    plt.plot(x, mean, marker=marker, color=color, linewidth=1.5)
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale('log')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")

def plot_two_with_errorbars(x, curves, xlabel, ylabel, title, path, logy=False):
    import matplotlib.pyplot as plt
    plt.figure()
    for label, (mean, std) in curves.items():
        plt.errorbar(x, mean, yerr=std, capsize=3, marker='o', label=label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    if logy: plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def save_results_txt(results, path, header=None):
    """Save a list of dict results as a human-readable text file.
    `header` is an optional dict with metadata to print at the top.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        if header is not None:
            f.write("# METADATA\n")
            for k, v in header.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        if len(results) == 0:
            f.write("<no results>\n")
            return

        keys = list(results[0].keys())
        f.write('\t'.join(keys) + '\n')
        for r in results:
            f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')

    print(f"Saved {path}")


def save_results_csv(results, path):
    """Save list-of-dicts as CSV. Field order follows the first dict's keys."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(results) == 0:
        with open(path, 'w') as f:
            f.write('')
        print(f"Saved {path} (empty)")
        return
    keys = list(results[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            # ensure keys present
            row = {k: r.get(k, '') for k in keys}
            writer.writerow(row)
    print(f"Saved {path}")


def save_results_json(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}")

def plot_full_spectrum_with_top_abs(evals, title, path, k_top=40, bins=120):
    import matplotlib.pyplot as plt

    # --- robust conversion ---
    if evals is None:
        raise ValueError(f"[plot_full_spectrum_with_top_abs] evals is None for: {title}")

    if hasattr(evals, "detach"):  # torch tensor
        evals = evals.detach().cpu().numpy()

    evals = np.asarray(evals)

    if evals.ndim == 0:
        raise ValueError(f"[plot_full_spectrum_with_top_abs] evals is scalar/0D for: {title} -> {evals}")

    evals = evals.reshape(-1)

    plt.figure(figsize=(7,4))
    plt.hist(evals, bins=bins, alpha=0.85)

    k = min(k_top, evals.size)
    top = evals[np.argsort(np.abs(evals))[-k:]]
    for v in top:
        plt.axvline(v, linewidth=0.35, linestyle="dashed")

    plt.title(title)
    plt.xlabel("eigenvalue (dashed = top |eigs|)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


