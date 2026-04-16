import math
import torch
import numpy as np
import warnings
import teacher

### helpers ###

def sym(A):
    return 0.5 * (A + A.transpose(-1, -2))

# Toggle: if True the frob_orthonormalize will fall back to an eigendecomposition
# when Cholesky fails (robust for near-singular Grams). If you prefer the previous
# behaviour (Cholesky only and let it raise on non-PD input) set this to False.
FROB_ORTHONORMALIZE_USE_FALLBACK = True

### Estimation with sequential method ###

@torch.no_grad()
def C_apply(stream_fn, A_list, d, n_total, device=None, dtype=torch.float32, input_mode="true"):
    """
    Inputs:
      A_list: (p,d,d) symmetric test matrices A^(1),...,A^(p)
      stream_fn(): yields batches (X,y) with X:(bs,d), y:(bs,)
    Output:
      C[A_list]: (p,d,d)
    Implements:
      C[A] = (1/(2n)) sum_mu y_mu * ( <tildeX_mu,A>*tildeX_mu - 2A )
      with tildeX_mu = x_mu x_mu^T - I
    """
    if device is None:
        device = A_list.device
    A_list = sym(A_list).to(device=device, dtype=dtype)

    if input_mode == "true":
        I = torch.eye(d, device=device, dtype=dtype)

        # precompute Tr(A) for <tildeX, A> = x^T A x - Tr(A)
        trA = torch.diagonal(A_list, dim1=-2, dim2=-1).sum(-1)  # (p,)

        Sum_term = torch.zeros_like(A_list, dtype=torch.float64)   # accumulates sum y <tildeX,A> tildeX
        Sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_seen = 0

        for X, y in stream_fn():
            X = X.to(device=device, dtype=dtype)   # (bs,d)
            y = y.to(device=device, dtype=dtype)   # (bs,)
            n_seen += X.shape[0]

            # tildeX_mu = x x^T - I  (not stored fully; we use identities)
            # scalar coeff: <tildeX_mu, A_k> = x^T A_k x - Tr(A_k) for k in [p]
            xAx = torch.einsum("bd,kde,be->bk", X, A_list, X)      # (bs,p)
            coeff = xAx - trA[None, :]                              # (bs,p)

            # For each k, accumulate sum_mu y_mu * coeff_mu,k * tildeX_mu
            # tildeX_mu contributes via x x^T and -I:
            # sum y*coeff*(x x^T)  minus  (sum y*coeff)*I
            ycoeff = y[:, None] * coeff                              # (bs,k)

            sum_ycoeff_xxt = torch.einsum("bd,bk,be->kde", X, ycoeff, X)  # (k,d,d)
            sum_ycoeff = ycoeff.sum(dim=0).to(torch.float64)              # (k,)

            Sum_term += sum_ycoeff_xxt.to(torch.float64) - sum_ycoeff[:,None,None]*I.to(torch.float64)
            Sum_y += y.sum().to(torch.float64)

        # Now assemble C[A]:
        # C[A] = (1/(2n)) ( Sum_term  - 2 * (sum_mu y_mu) * A )
        denom = float(n_seen) if n_seen > 0 else float(n_total)
        CA = (Sum_term - 2.0*Sum_y*A_list.to(torch.float64)) / (2.0*denom)
        return sym(CA)
    
    elif input_mode == "gauss":
        # --- flatten A_list into feature vectors a_k in R^m
        Aflat = teacher.flatten_A_sym_for_H2_feature(A_list)         # (p,m)
        Aflat = Aflat.to(device=device, dtype=dtype)

        Sum_term = torch.zeros_like(Aflat, dtype=torch.float64)  # (p,m)
        Sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_seen = 0

        for Z, y in stream_fn():
            if Z is None or y is None:
                raise RuntimeError("stream_fn yielded None in gauss mode (should stop instead).")
            Z = Z.to(device=device, dtype=dtype)   # (bs,m)
            y = y.to(device=device, dtype=dtype)   # (bs,)
            n_seen += Z.shape[0]

            coeff = Z @ Aflat.T                    # (bs,p)
            ycoeff = y[:, None] * coeff            # (bs,p)
            Sum_term += torch.einsum("bm,bp->pm", Z, ycoeff).to(torch.float64)
            Sum_y += y.sum().to(torch.float64)

        denom = float(n_seen) if n_seen > 0 else float(n_total)
        CAflat = (Sum_term - 2.0*Sum_y*Aflat.to(torch.float64)) / (2.0*denom)  # (p,m)

        # --- unflatten back to symmetric matrices (p,d,d)
        CA = teacher.unflatten_A_sym_from_H2_feature(CAflat.to(dtype=dtype), d=d)      # (p,d,d)
        return sym(CA)

# @torch.no_grad()
# def frob_orthonormalize(Q):
#     # Q: (k,d,d) -> orthonormal in Frobenius via QR on flattened columns
#     k,d,_ = Q.shape
#     Q = sym(Q)
#     M = Q.reshape(k, d*d).T   # (d*d, k)
#     Z, _ = torch.linalg.qr(M, mode="reduced")
#     return sym(Z.T.reshape(k,d,d))

@torch.no_grad()
def frob_orthonormalize(Q, eps=1e-12, use_qr_fallback=True):
    """
    Orthonormalize k symmetric matrices Q_k in Frobenius inner product.

    Primary path: Gram + (regularized) Cholesky
    Fallback: QR on flattened vectors (more robust than eigendecomp(G))
    """
    Q = sym(Q).to(torch.float64)                 # (k,d,d)
    k, d, _ = Q.shape

    # Fail fast on NaN/Inf
    if not torch.isfinite(Q).all():
        raise RuntimeError("frob_orthonormalize: non-finite entries in Q (NaN/Inf).")

    G = torch.einsum("kij,lij->kl", Q, Q)        # (k,k)
    G = 0.5 * (G + G.T)

    I = torch.eye(k, device=G.device, dtype=G.dtype)

    # relative regularization scale (helps a lot)
    scale = (torch.trace(G).abs() / k).item()
    reg = float(eps) * (scale + 1.0)

    try:
        L = torch.linalg.cholesky(G + reg * I)
        Linv = torch.linalg.solve_triangular(
            L, torch.eye(k, device=Q.device, dtype=Q.dtype), upper=False
        )
        Qo = torch.einsum("kl,lij->kij", Linv, Q)
        return sym(Qo).to(torch.float32)

    except RuntimeError:
        if not use_qr_fallback:
            raise

        # --- QR fallback on flattened representation
        M = Q.reshape(k, d * d).T                # (d^2, k)
        Qorth, _ = torch.linalg.qr(M, mode="reduced")  # (d^2, k)
        Qo = Qorth.T.reshape(k, d, d)
        return sym(Qo).to(torch.float32)

@torch.no_grad()
def row_orthonormalize(Q: torch.Tensor):
    """
    Q: (k,m) -> rows orthonormal
    """
    # QR on transpose
    Qt = Q.T.to(torch.float64)           # (m,k)
    Qr, _ = torch.linalg.qr(Qt)          # orthonormal columns
    return Qr.T.to(Q.dtype).contiguous() # (k,m)


@torch.no_grad()
def C_apply_vec(stream_fn, V, m, n_total, device=None, dtype=torch.float64):
    """
    Apply C = E[y Z Z^T] - E[y] I on a list of vectors V (k,m).
    stream_fn yields (Z,y) with Z:(bs,m), y:(bs,)
    """
    if device is None:
        device = V.device

    Vd = V.to(device=device, dtype=dtype)         # (k,m)
    Sum = torch.zeros_like(Vd, dtype=dtype)       # accumulates sum y (Z·v) Z
    Sum_y = torch.tensor(0.0, device=device, dtype=dtype)

    for Z, y in stream_fn():
        Zd = Z.to(device=device, dtype=dtype)     # (bs,m)
        yd = y.to(device=device, dtype=dtype)     # (bs,)
        Sum_y += yd.sum()

        coeff = Zd @ Vd.T                         # (bs,k)
        Sum += (yd[:, None] * coeff).T @ Zd       # (k,m)

    mean_y = Sum_y / float(n_total)
    return (Sum / float(n_total)) - mean_y * Vd   # (k,m)


@torch.no_grad()
def top_p_eigvecs_of_C_vec(stream_fn_factory, m, n_total, p, n_iter=8, oversamp=10, device=None):
    """
    Power iteration to get top p eigenvectors (by lambda, not abs).
    Returns Q[:p] shape (p,m) rows orthonormal.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = p + oversamp
    Q = torch.randn(k, m, device=device, dtype=torch.float32)
    Q = row_orthonormalize(Q)

    for _ in range(n_iter):
        CQ = C_apply_vec(stream_fn_factory(), Q, m=m, n_total=n_total, device=device)
        Q = row_orthonormalize(CQ.to(torch.float32))

    return Q[:p]


@torch.no_grad()
def top_p_eigvecs_abs_of_C_vec(stream_fn_factory, m: int, n_total: int, p: int,
                               n_iter: int = 10, oversamp: int = 10, device=None):
    """
    Power iteration on C_hat (vector form).
    Returns Q[:p] where Q rows approximate eigenvectors corresponding to largest |eigs|.
    Note: power iteration targets large magnitude modes; sign doesn't matter.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = p + oversamp
    Q = torch.randn(k, m, device=device)
    Q = row_orthonormalize(Q)

    for _ in range(n_iter):
        CQ = C_apply_vec(stream_fn_factory(), Q, m=m, n_total=n_total, device=device)
        Q = row_orthonormalize(CQ)

    return Q[:p]  # (p,m)


@torch.no_grad()
def ritz_eigs_of_C_vec(stream_fn_factory, m: int, n_total: int,
                       k: int = 200, n_iter: int = 12, oversamp: int = 20, device=None):
    """
    Returns approximate eigenvalues via Rayleigh-Ritz on span(Q):
      - build Q by power iterations
      - form small matrix R = Q C(Q)^T  (k x k)
      - eigvals(R) approximate leading eigenvalues in that subspace
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kk = k + oversamp
    Q = torch.randn(kk, m, device=device)
    Q = row_orthonormalize(Q)

    for _ in range(n_iter):
        CQ = C_apply_vec(stream_fn_factory(), Q, m=m, n_total=n_total, device=device)
        Q = row_orthonormalize(CQ)

    # final CQ for Rayleigh matrix
    CQ = C_apply_vec(stream_fn_factory(), Q, m=m, n_total=n_total, device=device)
    # R_ij = <Q_i, C(Q_j)>
    R = (Q @ CQ.T).to(torch.float64)  # (kk,kk)

    evals = torch.linalg.eigvalsh(R).cpu().numpy()
    evals.sort()
    return evals


@torch.no_grad()
def full_eigs_from_stream_if_small(stream_fn_factory, m: int, n_total: int, device=None):
    """
    Builds explicit matrix Mhat if m is small enough. O(m^2) memory.
    Mhat = (1/2n) sum y (z z^T - I)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M = torch.zeros((m, m), device=device, dtype=torch.float64)
    seen = 0
    for z, y in stream_fn_factory()():
        z = z.to(device=device, dtype=torch.float64)
        y = y.to(device=device, dtype=torch.float64)
        bs = z.shape[0]
        seen += bs

        # sum over batch: y * z z^T
        # (m,bs) * (bs,m) -> (m,m)
        M += (z.T * y[None, :]) @ z
        M -= y.sum() * torch.eye(m, device=device, dtype=torch.float64)

        if seen >= n_total:
            break

    M = M / (2.0 * float(n_total))
    evals = torch.linalg.eigvalsh(M).cpu().numpy()
    evals.sort()
    return evals


# @torch.no_grad()
# def top_p_eigmats_of_C(stream_fn_factory, d, n_total, p, n_iter=5, oversamp=3, device=None, input_mode='true'):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     k = p + oversamp
#     Q = torch.randn(k, d, d, device=device)
#     Q = frob_orthonormalize(Q)

#     for _ in range(n_iter):
#         CQ = C_apply(stream_fn_factory(), Q, d=d, n_total=n_total, device=device, input_mode=input_mode)
#         Q = frob_orthonormalize(CQ.to(dtype=torch.float32))

#     # return first p directions (already orthonormal)
#     return Q[:p]

@torch.no_grad()
def top_p_eigmats_of_C(
    stream_fn_factory,
    d,
    n_total,
    p,
    n_iter=5,
    oversamp=3,
    device=None,
    input_mode='true',
    Q_init=None,
    T_min=0,
    stop_tol=None,
    return_Q_full=False,
    return_info=False,
):
    """Estimate top-p eigen-matrices of the empirical operator C using (subspace) power iteration.

    IMPORTANT:
      - This routine is *operator-only*: it never forms C explicitly.
      - Each iteration calls C_apply(...) which streams over n_total samples.

    Added features (backwards compatible):
      - warm start via Q_init
      - early stopping on the *iterations* (never on samples) via (T_min, stop_tol)

    Returns:
      - default: Q[:p] of shape (p,d,d)
      - if return_Q_full: also returns Q_full of shape (k,d,d)
      - if return_info: also returns info dict with n_iter_effective and stop_delta
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = int(p) + int(oversamp)
    d = int(d)
    p = int(p)
    T_max = int(n_iter)
    T_min = int(T_min)

    # --- init subspace ---
    if Q_init is None:
        Q = torch.randn(k, d, d, device=device, dtype=torch.float32)
    else:
        Q0 = sym(Q_init.to(device=device, dtype=torch.float32))
        if Q0.dim() != 3 or Q0.shape[1] != d or Q0.shape[2] != d:
            raise ValueError(f"Q_init must have shape (k,d,d) with d={d}. Got {tuple(Q0.shape)}")
        if Q0.shape[0] < k:
            pad = torch.randn(k - Q0.shape[0], d, d, device=device, dtype=torch.float32)
            Q = torch.cat([Q0, pad], dim=0)
        else:
            Q = Q0[:k]

    Q = frob_orthonormalize(Q)  # (k,d,d) float32

    # --- early stop bookkeeping (subspace change between iterates) ---
    prev_Qp_flat = None
    last_delta = None
    n_iter_effective = 0

    for t in range(T_max):
        CQ = C_apply(stream_fn_factory(), Q, d=d, n_total=n_total, device=device, input_mode=input_mode)
        Q = frob_orthonormalize(CQ.to(dtype=torch.float32))
        n_iter_effective = t + 1

        if stop_tol is not None:
            # Compare p-dim subspaces spanned by current and previous iterates.
            Qp_flat = Q[:p].reshape(p, d * d).to(torch.float64)
            if prev_Qp_flat is not None:
                # Both bases are Frobenius-orthonormal => dot products form a p×p matrix.
                M = Qp_flat @ prev_Qp_flat.T  # (p,p)
                I = torch.eye(p, device=M.device, dtype=M.dtype)
                delta = torch.linalg.norm(I - (M @ M.T), ord='fro').item()
                last_delta = float(delta)
                if (t + 1) >= max(T_min, 1) and delta < float(stop_tol):
                    break
            prev_Qp_flat = Qp_flat.detach()

    out_Qp = Q[:p]
    info = {
        "n_iter_effective": int(n_iter_effective),
        "stop_delta": None if last_delta is None else float(last_delta),
        "T_max": int(T_max),
        "T_min": int(T_min),
        "stop_tol": None if stop_tol is None else float(stop_tol),
        "k": int(k),
    }

    if not return_Q_full and not return_info:
        return out_Qp
    if return_Q_full and not return_info:
        return out_Qp, Q
    if (not return_Q_full) and return_info:
        return out_Qp, info
    return out_Qp, Q, info

@torch.no_grad()
def compute_hhat_from_X_and_Ahat(X, Ahat):
    """
    X:   (bs, d)
    Ahat:(p, d, d)  symmetric
    returns hhat: (bs, p) with
        hhat_{mu,i} = <Ahat_i, H2(x_mu)> / sqrt(2)
                    = (x^T Ahat_i x - Tr(Ahat_i)) / sqrt(2)
    """
    # traces Tr(Ahat_i)
    trA = torch.diagonal(Ahat, dim1=-2, dim2=-1).sum(-1)              # (p,)
    # quadratic forms x^T A_i x
    q = torch.einsum("bd,pde,be->bp", X, Ahat, X)                     # (bs,p)
    hhat = (q - trA[None, :]) / math.sqrt(2.0)
    return hhat

def _apply_rf_activation(U: torch.Tensor, rf_activation: str):
    rf_activation = str(rf_activation)
    if rf_activation in {"id", "identity", "linear"}:
        return U
    if rf_activation == "relu":
        return torch.relu(U) - (1.0 / math.sqrt(2.0 * math.pi)) - 0.5 * U
    if rf_activation == "tanh":
        return torch.tanh(U)
    if rf_activation == "erf":
        return torch.erf(U)
    raise ValueError(f"Unknown rf_activation: {rf_activation}")


@torch.no_grad()
def init_rf_layer(
    d: int,
    rf_width: int,
    rf_activation: str = "relu",
    seed: int = 0,
    device=None,
    dtype=torch.float32,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    W = torch.randn(int(rf_width), int(d), generator=gen, device=device, dtype=dtype)
    W = W / math.sqrt(float(d))
    return {"W": W, "rf_activation": str(rf_activation)}


@torch.no_grad()
def apply_rf_layer(X: torch.Tensor, rf_layer: dict, device=None, dtype=torch.float32):
    if device is None:
        device = X.device
    X = X.to(device=device, dtype=dtype)
    W = rf_layer["W"].to(device=device, dtype=dtype)
    U = X @ W.T
    return _apply_rf_activation(U, rf_layer["rf_activation"])


@torch.no_grad()
def compute_hhat_from_X_and_rf(
    X: torch.Tensor,
    rf_layer: dict,
    Vhat: torch.Tensor,
    device=None,
    dtype=torch.float32,
):
    S = apply_rf_layer(X, rf_layer=rf_layer, device=device, dtype=dtype)  # (bs, rf_width)
    Vhat = Vhat.to(device=S.device, dtype=S.dtype)                         # (p, rf_width)
    return S @ Vhat.T                                                     # (bs, p)


@torch.no_grad()
def compute_hhat_from_X_and_rf_whitened(
    X: torch.Tensor,
    rf_layer: dict,
    Vhat: torch.Tensor,
    whiten_mu_cpu: torch.Tensor,
    whiten_mat_cpu: torch.Tensor,
    device=None,
    dtype=torch.float32,
):
    Hraw = compute_hhat_from_X_and_rf(
        X, rf_layer=rf_layer, Vhat=Vhat, device=device, dtype=dtype
    )
    Hwhite = apply_whitening_to_H(
        Hraw, whiten_mu_cpu, whiten_mat_cpu, device=device, dtype=dtype
    )
    return Hwhite

@torch.no_grad()
def estimate_whitening_from_H_stream(
    stream_fn,
    p: int,
    device=None,
    eps: float = 1e-6,
):
    """
    Given a stream yielding (H, y) with H shape (bs,p), estimate:
      mu   = E[H]
      Sigma= Cov(H)
      W    = Sigma^{-1/2}
    Returns CPU tensors: mu_cpu, Sigma_cpu, Wwhite_cpu
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- pass 1: mean ----
    h_sum = torch.zeros(p, device=device, dtype=torch.float32)
    count = 0
    for H, _ in stream_fn():
        H = H.to(device=device, dtype=torch.float32)
        h_sum += H.sum(dim=0)
        count += H.shape[0]
    if count == 0:
        raise ValueError("estimate_whitening_from_H_stream: empty stream")
    mu = h_sum / float(count)

    # ---- pass 2: covariance ----
    Sigma = torch.zeros((p, p), device=device, dtype=torch.float32)
    for H, _ in stream_fn():
        H = H.to(device=device, dtype=torch.float32)
        Hc = H - mu[None, :]
        Sigma += Hc.T @ Hc
    Sigma /= float(count)

    # regularized inverse square root
    evals, evecs = torch.linalg.eigh(Sigma)
    evals = torch.clamp(evals, min=float(eps))
    Winvhalf = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    Winvhalf = 0.5 * (Winvhalf + Winvhalf.T)

    return mu.detach().cpu(), Sigma.detach().cpu(), Winvhalf.detach().cpu()


@torch.no_grad()
def apply_whitening_to_H(
    H: torch.Tensor,
    mu_cpu: torch.Tensor,
    Wwhite_cpu: torch.Tensor,
    device=None,
    dtype=torch.float32,
):
    if device is None:
        device = H.device
    H = H.to(device=device, dtype=dtype)
    mu = mu_cpu.to(device=device, dtype=dtype)
    Wwhite = Wwhite_cpu.to(device=device, dtype=dtype)
    Hc = H - mu[None, :]
    return Hc @ Wwhite.T

@torch.no_grad()
def fit_rf_spectral_layer1_from_stream(
    stream_fn_factory,
    d: int,
    rf_width: int,
    p_out: int,
    n_total: int,
    rf_activation: str = "relu",
    rf_seed: int = 0,
    n_iter: int = 10,
    oversamp: int = 10,
    device=None,
    Q_init=None,
    T_min: int = 0,
    stop_tol=None,
    return_Q_full: bool = False,
    return_info: bool = False,
):
    """
    Fit top-p_out eigenvectors of
        C_rf = (1/n) sum_mu y_mu s_mu s_mu^T - mean(y) I
    where s_mu = sigma(W x_mu), W random frozen.

    Returns:
      rf_layer, Vhat
    and optionally Q_full, info
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rf_width = int(rf_width)
    p_out = int(p_out)
    k = p_out + int(oversamp)
    T_max = int(n_iter)
    T_min = int(T_min)

    rf_layer = init_rf_layer(
        d=int(d),
        rf_width=rf_width,
        rf_activation=rf_activation,
        seed=int(rf_seed),
        device=device,
        dtype=torch.float32,
    )

    if Q_init is None:
        Q = torch.randn(k, rf_width, device=device, dtype=torch.float32)
    else:
        Q0 = Q_init.to(device=device, dtype=torch.float32)
        if Q0.dim() != 2 or Q0.shape[1] != rf_width:
            raise ValueError(f"Q_init must have shape (k, rf_width={rf_width}). Got {tuple(Q0.shape)}")
        if Q0.shape[0] < k:
            pad = torch.randn(k - Q0.shape[0], rf_width, device=device, dtype=torch.float32)
            Q = torch.cat([Q0, pad], dim=0)
        else:
            Q = Q0[:k]
    Q = row_orthonormalize(Q)

    def rf_stream():
        for X, y in stream_fn_factory()():
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            S = apply_rf_layer(X, rf_layer=rf_layer, device=device, dtype=torch.float32)
            yield S, y

    prev_Qp = None
    last_delta = None
    n_iter_effective = 0

    for t in range(T_max):
        CQ = C_apply_vec(rf_stream, Q, m=rf_width, n_total=n_total, device=device)
        Q = row_orthonormalize(CQ.to(torch.float32))
        n_iter_effective = t + 1

        if stop_tol is not None:
            Qp = Q[:p_out].to(torch.float64)
            if prev_Qp is not None:
                M = Qp @ prev_Qp.T
                I = torch.eye(p_out, device=M.device, dtype=M.dtype)
                delta = torch.linalg.norm(I - (M @ M.T), ord="fro").item()
                last_delta = float(delta)
                if (t + 1) >= T_min and delta <= float(stop_tol):
                    break
            prev_Qp = Qp.clone()

    Vhat = Q[:p_out].contiguous()
    info = {
        "n_iter_effective": int(n_iter_effective),
        "stop_delta": None if last_delta is None else float(last_delta),
        "loaded_Ahat": False,
    }

    outs = [rf_layer, Vhat]
    if return_Q_full:
        outs.append(Q.contiguous())
    if return_info:
        outs.append(info)
    return tuple(outs) if len(outs) > 1 else outs[0]

@torch.no_grad()
def estimate_Bhat_from_stream(stream_fn, Ahat, p, n_total, device=None, dtype=torch.float32):
    """
    Implements Eq.(15):
        Bhat = (1/(2n)) sum_mu y_mu * (hhat_mu hhat_mu^T - I_p)
    streaming + GPU.

    Inputs:
      stream_fn(): yields batches (X,y), with X:(bs,d), y:(bs,)
      Ahat: (p,d,d) estimated matrices from step 1
      p: number of features
      n_total: total number of samples n
    Returns:
      Bhat (p,p) on CPU float64
    """
    if device is None:
        device = Ahat.device

    Ahat = Ahat.to(device=device, dtype=dtype)
    I_p = torch.eye(p, device=device, dtype=dtype)

    Sum_y_hhT = torch.zeros((p, p), device=device, dtype=torch.float64)  # sum y h h^T
    Sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)        # sum y

    for X, y in stream_fn():
        X = X.to(device=device, dtype=dtype)        # (bs,d)
        y = y.to(device=device, dtype=dtype)        # (bs,)

        hhat = compute_hhat_from_X_and_Ahat(X, Ahat)                 # (bs,p)

        # accumulate sum y * (h h^T)
        Sum_y_hhT += (hhat.T @ (hhat * y[:, None])).to(torch.float64)
        Sum_y += y.sum().to(torch.float64)

    # (1/(2n)) [ sum y h h^T  - (sum y) I ]
    Bhat = (Sum_y_hhT - Sum_y * I_p.to(torch.float64)) / (2.0 * float(n_total))
    Bhat = 0.5 * (Bhat + Bhat.T)  # symmetrize
    return Bhat.cpu()


@torch.no_grad()
def predict_y_from_Bhat_and_Ahat(X, Ahat, Bhat_cpu, device=None, dtype=torch.float32):
    """
    Given new X (bs,d), predict:
        yhat = <Bhat, H2(hhat)> = hhat^T Bhat hhat - Tr(Bhat)
    """
    if device is None:
        device = Ahat.device

    X = X.to(device=device, dtype=dtype)
    Ahat = Ahat.to(device=device, dtype=dtype)
    Bhat = Bhat_cpu.to(device=device, dtype=dtype)

    hhat = compute_hhat_from_X_and_Ahat(X, Ahat)  # (bs,p)
    yhat = torch.einsum("bp,pq,bq->b", hhat, Bhat, hhat) - torch.trace(Bhat)
    return yhat



import numpy as _np

def fit_polynomial_link(y0, y, degree=5, ridge=1e-6):
    """
    Fit y ≈ poly( (y0-mu)/sig ) with coeffs in increasing order:
      c0 + c1 x + ... + cD x^D
    Returns: coeffs, (mu, sig)
    """
    y0 = _np.asarray(y0, dtype=_np.float64).reshape(-1)
    y  = _np.asarray(y,  dtype=_np.float64).reshape(-1)

    mu = float(y0.mean())
    sig = float(y0.std())
    if sig < 1e-12:
        coeffs = _np.zeros(degree + 1, dtype=_np.float64)
        coeffs[0] = float(y.mean())
        return coeffs, (mu, sig)

    x = (y0 - mu) / (sig + 1e-12)
    X = _np.vander(x, N=degree + 1, increasing=True)

    XtX = X.T @ X
    scale = float(_np.trace(XtX) / XtX.shape[0])
    lam = float(ridge) * (scale + 1.0)
    A = XtX + lam * _np.eye(XtX.shape[0])
    b = X.T @ y

    try:
        coeffs = _np.linalg.solve(A, b)
    except _np.linalg.LinAlgError:
        coeffs, *_ = _np.linalg.lstsq(X, y, rcond=None)

    return coeffs, (mu, sig)

@torch.no_grad()
def fit_affine_link(s_train, y_train):
    """
    Fit y ≈ a*s + b by least squares.
    Inputs are 1D numpy arrays.
    Returns floats (a, b).
    """
    s = np.asarray(s_train, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)

    X = np.column_stack([s, np.ones_like(s)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    return float(a), float(b)


@torch.no_grad()
def predict_affine_link(s, a, b):
    s = np.asarray(s, dtype=np.float64)
    return a * s + b


@torch.no_grad()
def estimate_Bhat_from_H_stream(stream_fn, p, n_total, device=None, dtype=torch.float32):
    """
    Estimate Bhat when the stream yields (H, y) directly where H: (bs,p).
    Implements Bhat = (1/(2n)) sum_mu y_mu * (h_mu h_mu^T - I_p)
    Returns Bhat (p,p) on CPU float64.
    """
    if device is None:
        # try to infer device from first yield
        try:
            it = iter(stream_fn())
            H0, y0 = next(it)
            device = H0.device
        except Exception:
            device = torch.device('cpu')

    I_p = torch.eye(p, device=device, dtype=dtype)
    Sum_y_hhT = torch.zeros((p, p), device=device, dtype=torch.float64)
    Sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)

    for H, y in stream_fn():
        H = H.to(device=device, dtype=dtype)   # (bs,p)
        y = y.to(device=device, dtype=dtype)   # (bs,)

        Sum_y_hhT += (H.T @ (H * y[:, None])).to(torch.float64)
        Sum_y += y.sum().to(torch.float64)

    Bhat = (Sum_y_hhT - Sum_y * I_p.to(torch.float64)) / (2.0 * float(n_total))
    Bhat = 0.5 * (Bhat + Bhat.T)
    return Bhat.cpu()

@torch.no_grad()
def estimate_Bhat_from_H_stream_nonisotropic(
    stream_fn,
    p: int,
    n_total: int,
    device=None,
    return_stats: bool = False,
):
    """
    Estimate
        Bhat = (1/n) sum_mu y_mu [ (h_mu - m)(h_mu - m)^T - Sigma ]
    where
        m     = (1/n) sum_mu h_mu
        Sigma = (1/n) sum_mu (h_mu - m)(h_mu - m)^T

    stream_fn must yield (H_batch, y_batch), with H_batch shape (bs, p), y_batch shape (bs,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- pass 1: mean of h ----
    h_sum = torch.zeros(p, device=device, dtype=torch.float32)
    count = 0
    for H, _ in stream_fn():
        H = H.to(device=device, dtype=torch.float32)
        h_sum += H.sum(dim=0)
        count += H.shape[0]
    if count == 0:
        raise ValueError("estimate_Bhat_from_H_stream_nonisotropic: empty stream")
    h_mean = h_sum / float(count)

    # ---- pass 2: covariance of h and weighted centered second moment ----
    Sigma = torch.zeros((p, p), device=device, dtype=torch.float32)
    Bhat = torch.zeros((p, p), device=device, dtype=torch.float32)
    y_sum = 0.0
    y_count = 0

    for H, y in stream_fn():
        H = H.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        Hc = H - h_mean[None, :]

        Sigma += Hc.T @ Hc
        Bhat += Hc.T @ (Hc * y[:, None])

        y_sum += y.sum().item()
        y_count += y.numel()

    Sigma /= float(count)
    Bhat /= float(count)
    y_mean = y_sum / max(y_count, 1)

    Bhat = Bhat - float(y_mean) * Sigma
    Bhat = 0.5 * (Bhat + Bhat.T)

    Bhat_cpu = Bhat.detach().cpu()
    if not return_stats:
        return Bhat_cpu

    return (
        Bhat_cpu,
        h_mean.detach().cpu(),
        Sigma.detach().cpu(),
    )

def predict_polynomial_link(y0, coeffs, mu_sig=None):
    import numpy as _np
    if hasattr(y0, 'detach'):
        y0 = y0.detach().cpu().numpy().reshape(-1)
    else:
        y0 = _np.asarray(y0).reshape(-1)

    if mu_sig is not None:
        mu, sig = mu_sig
        y0 = (y0 - float(mu)) / (float(sig) + 1e-12)

    # Horner / increasing order: c0 + c1 x + ...
    y = _np.zeros_like(y0, dtype=_np.float64)
    xp = _np.ones_like(y0, dtype=_np.float64)
    for c in coeffs:
        y += c * xp
        xp *= y0
    return y

    
@torch.no_grad()
def collect_representation_train_from_stream(
    stream_fn_factory,
    representation: str,
    n_keep,
    model: str,
    Ahat=None,
    device=None,
    dtype=torch.float32,
):
    """
    Collect a train set (Phi_train, y_train) from the normalized stream.

    representation:
      - 'x'  : observable input (X in true mode, Z in gauss mode)
      - 'h1' : estimated first-layer features \hat h^{(1)}

    n_keep:
      - positive int  -> keep at most n_keep samples
      - None or <= 0  -> keep the FULL stream (no cap)

    Returns CPU float32 tensors.
    """
    representation = str(representation)
    if representation not in {"x", "h1"}:
        raise ValueError(f"Unknown representation={representation}")
    if representation == "h1" and Ahat is None:
        raise ValueError("Ahat must be provided when representation='h1'")

    if device is None:
        if Ahat is not None:
            device = Ahat.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_full_stream = (n_keep is None) or (int(n_keep) <= 0)
    if not use_full_stream:
        n_keep = int(n_keep)

    Ahat_dev = None
    Ahat_flat = None
    if representation == "h1":
        Ahat_dev = Ahat.to(device=device, dtype=dtype)
        if model == "gauss":
            Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat_dev).to(device=device, dtype=dtype)

    Phi_chunks, y_chunks = [], []
    kept = 0

    for X_or_Z, y in stream_fn_factory()():
        X_or_Z = X_or_Z.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        if representation == "x":
            Phi = X_or_Z
        else:
            if model == "true":
                Phi = compute_hhat_from_X_and_Ahat(X_or_Z, Ahat_dev)
            else:
                Phi = X_or_Z @ Ahat_flat.T

        if use_full_stream:
            Phi_chunks.append(Phi.detach().cpu())
            y_chunks.append(y.detach().cpu())
            kept += Phi.shape[0]
            continue

        take = min(Phi.shape[0], n_keep - kept)
        if take <= 0:
            break

        Phi_chunks.append(Phi[:take].detach().cpu())
        y_chunks.append(y[:take].detach().cpu())
        kept += take

        if kept >= n_keep:
            break

    if kept == 0:
        raise RuntimeError("No samples were collected for RBF-KRR")

    Phi_train = torch.cat(Phi_chunks, dim=0).to(torch.float32)
    y_train = torch.cat(y_chunks, dim=0).to(torch.float32)
    return Phi_train, y_train

@torch.no_grad()
def build_test_representation(
    X_or_Z_test,
    representation: str,
    model: str,
    Ahat=None,
    device=None,
    dtype=torch.float32,
):
    """
    Build Phi_test on CPU float32 from a test observable X_or_Z_test.
    """
    representation = str(representation)
    if representation not in {"x", "h1"}:
        raise ValueError(f"Unknown representation={representation}")
    if representation == 'h1' and Ahat is None:
        raise ValueError("Ahat must be provided when representation='h1'")

    if device is None:
        if Ahat is not None:
            device = Ahat.device
        elif hasattr(X_or_Z_test, 'device'):
            device = X_or_Z_test.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_or_Z_test = X_or_Z_test.to(device=device, dtype=dtype)
    if representation == 'x':
        return X_or_Z_test.detach().cpu().to(torch.float32)

    Ahat_dev = Ahat.to(device=device, dtype=dtype)
    if model == 'true':
        Phi_test = compute_hhat_from_X_and_Ahat(X_or_Z_test, Ahat_dev)
    else:
        Ahat_flat = teacher.flatten_A_sym_for_H2_feature(Ahat_dev).to(device=device, dtype=dtype)
        Phi_test = X_or_Z_test @ Ahat_flat.T
    return Phi_test.detach().cpu().to(torch.float32)


@torch.no_grad()
def standardize_features_train_test(Phi_train, Phi_test, eps=1e-6):
    """
    Standardize each coordinate using train-set mean/std.
    Returns standardized train/test plus the train statistics.
    """
    Phi_train = Phi_train.to(torch.float32)
    Phi_test = Phi_test.to(torch.float32)

    mu = Phi_train.mean(dim=0, keepdim=True)
    std = Phi_train.std(dim=0, unbiased=False, keepdim=True)
    std = torch.clamp(std, min=float(eps))

    Phi_train_std = (Phi_train - mu) / std
    Phi_test_std = (Phi_test - mu) / std
    return Phi_train_std, Phi_test_std, mu, std


@torch.no_grad()
def median_heuristic_sigma(Phi_train, max_points=2000):
    """
    Classical median heuristic for RBF kernels:
      sigma = sqrt( median_{i<j} ||x_i - x_j||^2 )
    computed on a subsample when n is large.
    """
    Phi_train = Phi_train.to(torch.float32)
    n = int(Phi_train.shape[0])
    if n <= 1:
        return 1.0

    if n > int(max_points):
        idx = torch.randperm(n)[: int(max_points)]
        X = Phi_train[idx]
    else:
        X = Phi_train

    d2 = torch.cdist(X, X, p=2) ** 2
    iu = torch.triu_indices(d2.shape[0], d2.shape[1], offset=1)
    vals = d2[iu[0], iu[1]]
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return 1.0

    med = torch.median(vals).item()
    return float(max(med, 1e-12) ** 0.5)


@torch.no_grad()
def rbf_kernel(X1, X2, sigma: float):
    sigma = float(max(sigma, 1e-12))
    d2 = torch.cdist(X1, X2, p=2) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma))


@torch.no_grad()
def fit_rbf_krr(
    Phi_train,
    y_train,
    sigma=None,
    lam: float = 1e-4,
    sigma_mult: float = 1.0,
    device=None,
):
    """
    Exact kernel ridge regression with an RBF kernel:
      alpha = (K + lam * n I)^{-1} y
    Stores the train features and dual coefficients on CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Phi_train = Phi_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device, dtype=torch.float32).reshape(-1)
    n = int(Phi_train.shape[0])
    if n != int(y_train.shape[0]):
        raise ValueError(f"Mismatched sizes: {Phi_train.shape[0]} train points vs {y_train.shape[0]} labels")

    sigma_base = float(median_heuristic_sigma(Phi_train.detach().cpu())) if sigma is None else float(sigma)
    sigma_eff = float(max(1e-8, float(sigma_mult) * sigma_base))

    K = rbf_kernel(Phi_train, Phi_train, sigma_eff)
    Kreg = K + (float(lam) * float(n)) * torch.eye(n, device=device, dtype=torch.float32)

    try:
        L = torch.linalg.cholesky(Kreg)
        alpha = torch.cholesky_solve(y_train[:, None], L).squeeze(1)
    except RuntimeError:
        alpha = torch.linalg.solve(Kreg, y_train)

    return {
        'Phi_train': Phi_train.detach().cpu(),
        'alpha': alpha.detach().cpu(),
        'sigma': float(sigma_eff),
        'sigma_base': float(sigma_base),
        'lam': float(lam),
        'n_train_krr': int(n),
    }


@torch.no_grad()
def predict_rbf_krr(model, Phi_test, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Phi_train = model['Phi_train'].to(device=device, dtype=torch.float32)
    alpha = model['alpha'].to(device=device, dtype=torch.float32)
    sigma = float(model['sigma'])

    Phi_test = Phi_test.to(device=device, dtype=torch.float32)
    K_test = rbf_kernel(Phi_test, Phi_train, sigma)
    yhat = K_test @ alpha
    return yhat.detach().cpu()

### New Bhat for the 3 layers model ###

# @torch.no_grad()
# def estimate_Bhat_from_features_stream(stream_fn, p, n_total, device, dtype=torch.float32):
#     # stream_fn() yields (H2_hat, y)
#     Bsum = torch.zeros(p, p, device=device, dtype=torch.float64)
#     for H, y in stream_fn():
#         H = H.to(device=device, dtype=dtype)
#         y = y.to(device=device, dtype=dtype)
#         # H2(H) = (H H^T - I)/sqrt(2) per sample => Bhat accumulates y * H2(H)
#         # Here we build per-batch: sum_i y_i * (h_i h_i^T - I)
#         Bsum += torch.einsum("b,bi,bj->ij", y, H, H).to(torch.float64)
#     Bhat = (Bsum / max(n_total, 1)).cpu()
#     # subtract trace term is handled at prediction time by (h^T B h - tr B)
#     return Bhat

### Estimation with naive method ###

def He4_torch(t):
    t2 = t*t
    return t2*t2 - 6.0*t2 + 3.0

# Deprecated: naive order-2 / order-4 predictors were removed. If you need
# them for legacy experiments, they can be reintroduced from git history.


### Spectrum of the student features ###

@torch.no_grad()
def Ctilde_matvec_streaming(
    w,                       # (D,) on device
    stream_fn,               # callable: stream_fn() yields (X, y) with X (bs,d), y (bs,)
    d, n_total,
    device,
    dtype=torch.float32,
    batch_cap=None,          # optional: limit number of batches for speed
):
    """
    Computes (Ctilde @ w) without forming Ctilde (D=d^2).
    Ctilde = (1/(n*sqrt(2))) * sum_mu y_mu * (xt xt^T - I),
    xt = vec(H2(x)), H2(x) = (xx^T - I_d)/sqrt(2).
    """
    raise NotImplementedError("Ctilde_matvec_streaming was removed (SLQ support deprecated). Use dense_spectrum_from_stream or ritz_eigs_of_C instead.")


@torch.no_grad()
# SLQ / Lanczos-based spectral density estimation removed (deprecated)


# ---------- convenience wrapper for your Ctilde ----------
def make_Ctilde_matvec(stream_fn, d, n_total, device, batch_cap=None):
    raise NotImplementedError("make_Ctilde_matvec removed: SLQ-based spectral density estimation deprecated. Use dense_spectrum_from_stream or ritz_eigs_of_C.")

### The 8 spectrum check ###


@torch.no_grad()
def dense_spectrum_from_stream(
    stream_fn_factory,
    dim,
    n_total,
    device=None,
    dtype=torch.float32,
    return_evecs=False,
    center=True,          # centre le bulk en retirant trace/dim si tu veux (optionnel)
):
    """
    Build dense C = (1/n) sum y (z z^T - I) from stream, then compute full eigenspectrum.
    Returns:
      - evals: (dim,) all eigenvalues sorted ascending
      - (optional) evecs: (dim,dim) eigenvectors (columns)
      - C (optional): you can save it if needed (commented)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accumulate dense matrix in float64 for numerical stability
    C = torch.zeros((dim, dim), device=device, dtype=torch.float64)
    sum_y = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Stream batches
    for Z, y in stream_fn_factory()():   # IMPORTANT: factory returns a function, then call it
        Z = Z.to(device=device, dtype=dtype)        # (bs,dim)
        y = y.to(device=device, dtype=dtype)        # (bs,)

        # weighted Gram: Z^T diag(y) Z = ( (sqrt(|y|) Z)^T * sign(y) * (sqrt(|y|) Z) )
        # simplest: (Z.T @ (y[:,None] * Z))
        C += (Z.to(torch.float64).T @ ((y[:, None]).to(torch.float64) * Z.to(torch.float64)))
        sum_y += y.to(torch.float64).sum()

    C /= float(n_total)
    mean_y = sum_y / float(n_total)

    # subtract mean_y * I  because (1/n) sum y * (-I)
    C -= mean_y * torch.eye(dim, device=device, dtype=torch.float64)

    # Optional: recentre le bulk (souvent utile pour lisibilité)
    if center:
        C -= (torch.trace(C) / dim) * torch.eye(dim, device=device, dtype=torch.float64)

    # Full eigendecomposition (symmetric)
    if return_evecs:
        evals, evecs = torch.linalg.eigh(C)  # evals asc, evecs columns
        return {
            "evals": evals.to(torch.float64).cpu().numpy(),
            "evecs": evecs.to(torch.float64).cpu().numpy(),
            "mean_y": float(mean_y.cpu().item()),
        }
    else:
        evals = torch.linalg.eigvalsh(C)     # faster, no vectors
        return {
            "evals": evals.to(torch.float64).cpu().numpy(),
            "mean_y": float(mean_y.cpu().item()),
        }


# ============================================================
# Linear regression for skip connections: y ≈ h1·b1 + h2·b2
# Streaming least squares on GPU
# ============================================================

@torch.no_grad()
def fit_b1_b2_stream(
    stream_fn,              # yields (X, y)
    A1_hat, A2_hat,
    p1, p2, n_total,
    device=None,
    dtype=torch.float32,
    ridge=1e-8
):
    """
    Fit (b1,b2) by streaming least squares:
      minimize || y - h1_hat b1 - h2_hat b2 ||^2

    stream_fn(): yields batches (X,y)
    Returns b1_hat, b2_hat on CPU float64
    """
    if device is None:
        device = A1_hat.device

    A1_hat = A1_hat.to(device=device, dtype=dtype)
    A2_hat = A2_hat.to(device=device, dtype=dtype)

    P = p1 + p2
    XtX = torch.zeros((P, P), device=device, dtype=torch.float64)
    Xty = torch.zeros((P,), device=device, dtype=torch.float64)

    for X, y in stream_fn():
        X = X.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        h1 = compute_hhat_from_X_and_Ahat(X, A1_hat)     # (bs,p1)
        h2 = compute_hhat_from_X_and_Ahat(h1, A2_hat)    # (bs,p2)

        Phi = torch.cat([h1, h2], dim=1)                 # (bs,P)

        XtX += (Phi.T @ Phi).to(torch.float64)
        Xty += (Phi.T @ y).to(torch.float64)

    # ridge for stability
    XtX = XtX + ridge * torch.eye(P, device=device, dtype=torch.float64)

    w = torch.linalg.solve(XtX, Xty)  # (P,)
    b1 = w[:p1].cpu()
    b2 = w[p1:].cpu()
    return b1, b2


@torch.no_grad()
def predict_y_hier2_skip(
    X, A1_hat, A2_hat, b1_hat_cpu, b2_hat_cpu,
    device=None, dtype=torch.float32
):
    """
    Predict yhat = h1_hat·b1_hat + h2_hat·b2_hat
    """
    if device is None:
        device = A1_hat.device

    X = X.to(device=device, dtype=dtype)
    A1_hat = A1_hat.to(device=device, dtype=dtype)
    A2_hat = A2_hat.to(device=device, dtype=dtype)
    b1 = b1_hat_cpu.to(device=device, dtype=dtype)
    b2 = b2_hat_cpu.to(device=device, dtype=dtype)

    h1 = compute_hhat_from_X_and_Ahat(X, A1_hat)
    h2 = compute_hhat_from_X_and_Ahat(h1, A2_hat)
    yhat = (h1 @ b1) + (h2 @ b2)
    return yhat


@torch.no_grad()
def ritz_eigs_of_C(stream_fn_factory, d, n_total, k=40, n_iter=15, oversamp=10, device=None):
    """
    Approx top eigenvalues of the linear operator C via Ritz values on a Krylov subspace.
    Returns evals (k+oversamp,) sorted decreasing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ktot = k + oversamp
    Q = torch.randn(ktot, d, d, device=device)
    Q = frob_orthonormalize(Q)

    for _ in range(n_iter):
        CQ = C_apply(stream_fn_factory(), Q, d=d, n_total=n_total, device=device)
        Q = frob_orthonormalize(CQ.to(torch.float32))

    # final Rayleigh-Ritz matrix M_ij = <Q_i, C(Q_j)>_F
    CQ = C_apply(stream_fn_factory(), Q, d=d, n_total=n_total, device=device)
    Q64  = Q.to(torch.float64)
    CQ64 = CQ.to(torch.float64)
    M = torch.einsum("kij,lij->kl", Q64, CQ64)
    M = 0.5 * (M + M.T)

    evals = torch.linalg.eigvalsh(M).cpu().numpy()[::-1]
    return evals
