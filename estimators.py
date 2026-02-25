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


@torch.no_grad()
def top_p_eigmats_of_C(stream_fn_factory, d, n_total, p, n_iter=5, oversamp=3, device=None, input_mode='true'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = p + oversamp
    Q = torch.randn(k, d, d, device=device)
    Q = frob_orthonormalize(Q)

    for _ in range(n_iter):
        CQ = C_apply(stream_fn_factory(), Q, d=d, n_total=n_total, device=device, input_mode=input_mode)
        Q = frob_orthonormalize(CQ.to(dtype=torch.float32))

    # return first p directions (already orthonormal)
    return Q[:p]

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
