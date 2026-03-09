import math
import torch

# -----------------------
# Public defaults (keep runners simple)
# -----------------------

DEFAULT_A_MODE = "sym_orth_frob"
DEFAULT_BETA = 1.0
DEFAULT_EPS = 0.0
DEFAULT_GAMMA = 0.0
DEFAULT_B_MODE = "dense" 


# -----------------------
# PRIORS and TOOLKIT
# -----------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def orthonormal_U_torch(d, p, gen, device):
    G = torch.randn(d, p, generator=gen, device=device)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    return Q[:, :p]

def gen_A_rank1_orth_torch(d, p, gen, device):
    U = orthonormal_U_torch(d, p, gen, device)  # (d,p)
    return {"mode": "rank1_orth", "U": U}

def gen_A_sym_orth_frob_torch(d, p, gen, device):
    # Gram-Schmidt Frobenius on symmetric matrices
    A_list = []
    for _ in range(p):
        G = torch.randn(d, d, generator=gen, device=device)
        A = 0.5 * (G + G.T)
        for Aj in A_list:
            proj = torch.sum(A * Aj)
            A = A - proj * Aj
        A = A / (torch.linalg.norm(A) + 1e-12)
        A_list.append(A)
    A = torch.stack(A_list, dim=0)  # (p,d,d)
    return {"mode": "sym_orth_frob", "A": A}

def gen_B_symmetric_dense_torch(p, gen, device, beta=None):
    if beta is None:
        beta = DEFAULT_BETA
    G = torch.randn(p, p, generator=gen, device=device)
    B = 0.5 * (G + G.T)
    fro = torch.linalg.norm(B) + 1e-12
    B = B / (fro / math.sqrt(p))
    B = (beta / p) * B
    return B

def gen_B_powerlaw_diag_torch(p, gen, device, beta=None, gamma=0.0, rademacher=True, eps=1e-12):
    """
    Diagonal power-law prior:
      B = diag( b_i ),  b_i = (beta/p) * z_i * i^{-gamma}
    where z_i are Rademacher (+/-1) if rademacher else all +1.
    """
    if beta is None:
        beta = DEFAULT_BETA

    idx = torch.arange(1, p+1, device=device, dtype=torch.float32)  # 1..p
    w = idx.pow(-1.0 * float(gamma))  # i^{-2gamma}

    if rademacher:
        z = torch.randint(0, 2, (p,), generator=gen, device=device, dtype=torch.int64)
        z = (2*z - 1).to(torch.float32)  # +/-1
    else:
        z = torch.ones(p, device=device, dtype=torch.float32)

    b = z * w
    # scale comparable to dense case
    if gamma==0.0:
        b = (beta / p) * b
    elif 0 < gamma <= 1/2:
        b = p**(2*gamma - 1.0) * b
    elif 1/2 < gamma:
        b = beta * b
    return torch.diag(b)

def compute_h_from_X_torch(X, A_teacher):
    # X: (bs,d)
    mode = A_teacher["mode"]
    if mode == "rank1_orth":
        U = A_teacher["U"]          # (d,p)
        Z = X @ U                   # (bs,p)
        H = (Z**2 - 1.0) / math.sqrt(2.0)
        return H

    if mode == "sym_orth_frob":
        A = A_teacher["A"]          # (p,d,d)
        trA = torch.diagonal(A, dim1=1, dim2=2).sum(-1)  # (p,)
        quad = torch.einsum("bd,pde,be->bp", X, A, X)     # (bs,p)
        H = (quad - trA[None, :]) / math.sqrt(2.0)
        return H

    raise ValueError("Unknown A_teacher mode")

def compute_y_from_H_torch(H, B):
    """
    Compute scalar outputs y = h^T B h - Tr(B) for batch H (bs,p).
    If an activation g is desired on the scalar output, call the returned
    values through that function outside (or use stream helpers that accept g).
    """
    y = torch.einsum("bp,pq,bq->b", H, B, H) - torch.trace(B)
    return y


def get_activation_fn(g_name=None, g_callable=None):
    """
    Return a callable that maps a torch tensor to another tensor.
    Supports g_name in {'id','tanh','relu_centered'} or a user-provided g_callable.
    """
    if g_callable is not None:
        return g_callable
    if g_name is None or g_name == 'id':
        return lambda x: x
    if g_name == 'tanh':
        return lambda x: x.tanh()
    if g_name == 'relu_centered':
        import torch.nn.functional as F
        def relu_c(x):
            xr = F.relu(x)
            return xr - xr.mean(dim=0, keepdim=True)
        return relu_c
    raise ValueError(f"Unknown g_name: {g_name}")

def sym_triu_indices(d, device=None):
    return torch.triu_indices(d, d, offset=0, device=device)

def flatten_H2_of_X_sym_hermite(X):
    """
    X: (bs,d) standard Gaussian
    returns Z: (bs,m) with m=d(d+1)/2
    Z_{ij}=x_i x_j for i<j
    Z_{ii}=(x_i^2-1)/sqrt(2)
    so each coord has Var=1
    """
    bs, d = X.shape
    device = X.device
    idx = sym_triu_indices(d, device=device)
    i, j = idx[0], idx[1]

    # compute all x_i x_j on upper triangle
    Z = X[:, i] * X[:, j]                 # (bs,m)

    # fix diagonal scaling: (x_i^2-1)/sqrt2 instead of x_i^2
    diag_mask = (i == j)
    if diag_mask.any():
        Xi = X[:, i[diag_mask]]           # (bs,d)
        Z[:, diag_mask] = (Xi**2 - 1.0) / math.sqrt(2.0)

    return Z

def flatten_A_sym_for_H2_feature(A):
    """
    A: (p,d,d) symmetric
    returns Aflat: (p,m) so that <Aflat_i, Z> = (x^T A_i x - trA_i)/sqrt2
    mapping:
      offdiag i<j: sqrt2 * A_ij
      diag: A_ii
    """
    p, d, _ = A.shape
    device = A.device
    idx = sym_triu_indices(d, device=device)
    i, j = idx[0], idx[1]

    Aflat = A[:, i, j].clone()            # (p,m)
    diag_mask = (i == j)
    if diag_mask.any():
        # diag unchanged
        pass
    off_mask = ~diag_mask
    if off_mask.any():
        Aflat[:, off_mask] = math.sqrt(2.0) * Aflat[:, off_mask]

    return Aflat

def unflatten_A_sym_from_H2_feature(Aflat, d):
    """
    Inverse of flatten_A_sym_for_H2_feature.

    Aflat: (p,m) or (m,) with m=d(d+1)/2
    returns A: (p,d,d) symmetric such that flatten_A_sym_for_H2_feature(A)=Aflat
    mapping inverse:
      diag: A_ii = Aflat_ii
      offdiag i<j: A_ij = Aflat_ij / sqrt2
    """
    if Aflat.dim() == 1:
        Aflat = Aflat.unsqueeze(0)  # (1,m)

    p, m = Aflat.shape
    device = Aflat.device
    dtype = Aflat.dtype

    idx = sym_triu_indices(d, device=device)   # (2,m)
    i, j = idx[0], idx[1]

    A = torch.zeros((p, d, d), device=device, dtype=dtype)

    diag_mask = (i == j)
    off_mask = ~diag_mask

    # diag
    if diag_mask.any():
        A[:, i[diag_mask], j[diag_mask]] = Aflat[:, diag_mask]

    # offdiag
    if off_mask.any():
        vals = Aflat[:, off_mask] / math.sqrt(2.0)
        ii = i[off_mask]
        jj = j[off_mask]
        A[:, ii, jj] = vals
        A[:, jj, ii] = vals  # symmetrize

    return A


# -----------------------------
# H3 utilities (moved from teacher_h3.py)
# -----------------------------
_SYM3_CACHE = {}

def sym_triu_indices_3(d: int, device=None):
    key = (d, str(device))
    if key in _SYM3_CACHE:
        return _SYM3_CACHE[key]
    idx = torch.combinations(torch.arange(d, device=device), r=3, with_replacement=True)  # (m,3)
    _SYM3_CACHE[key] = idx
    return idx


@torch.no_grad()
def flatten_H3_sym(X: torch.Tensor, idx: torch.Tensor = None):
    """
    Flatten H3 symmetric Hermite features (k=3) into vector Z (bs,m) with Var=1 coords.
    This mirrors the implementation previously in `teacher_h3.py`.
    """
    bs, d = X.shape
    device = X.device
    dtype  = X.dtype

    if idx is None:
        idx = sym_triu_indices_3(d, device=device)  # (m,3), i<=j<=k

    i = idx[:, 0]
    j = idx[:, 1]
    k = idx[:, 2]
    m = idx.shape[0]

    out = torch.zeros((bs, m), device=device, dtype=dtype)

    eq_ij = (i == j)
    eq_jk = (j == k)

    three_eq     = eq_ij & eq_jk          # i=j=k
    two_eq_ij    = eq_ij & (~eq_jk)       # i=j<k
    two_eq_jk    = eq_jk & (~eq_ij)       # i<j=k
    all_distinct = (~eq_ij) & (~eq_jk)    # i<j<k (since i<=j<=k)

    # i<j<k : x_i x_j x_k
    if all_distinct.any():
        ii = i[all_distinct]; jj = j[all_distinct]; kk = k[all_distinct]
        out[:, all_distinct] = (X[:, ii] * X[:, jj] * X[:, kk]) * math.sqrt(6.0)

    # i=i<k : (x_i^2 - 1) x_k
    if two_eq_ij.any():
        ii = i[two_eq_ij]; kk = k[two_eq_ij]
        out[:, two_eq_ij] = ((X[:, ii]**2 - 1.0) * X[:, kk]) * math.sqrt(3.0)

    # i<j=j : x_i (x_j^2 - 1)
    if two_eq_jk.any():
        ii = i[two_eq_jk]; jj = j[two_eq_jk]
        out[:, two_eq_jk] = (X[:, ii] * (X[:, jj]**2 - 1.0)) * math.sqrt(3.0)

    # i=i=i : (x_i^3 - 3 x_i)
    if three_eq.any():
        ii = i[three_eq]
        out[:, three_eq] = (X[:, ii]**3 - 3.0 * X[:, ii]) * 1.0

    # global factor (analogue au /sqrt(2) du cas H2)
    out = out / math.sqrt(6.0)

    return out


def flatten_Hk(X: torch.Tensor, k: int, idx: torch.Tensor = None):
    """
    Generic flatten for Hermite-k features. Supported k: 2, 3.
    - k=2 -> calls `flatten_H2_of_X_sym_hermite`
    - k=3 -> calls `flatten_H3_sym`
    """
    if k == 2:
        return flatten_H2_of_X_sym_hermite(X)
    elif k == 3:
        return flatten_H3_sym(X, idx=idx)
    else:
        raise ValueError(f"flatten_Hk only supports k=2 or k=3 (got k={k})")


def stream_batches_teacher_k(d: int, p: int, n: int, batch_size: int, k: int = 2, A_mode=None, beta=None, seed=0, device=None, normalize_y=True, return_params=True, input_mode='true', gamma=DEFAULT_GAMMA):
    """
    Generic stream factory for k=2 or k=3. For k=2 it calls `stream_batches_teacher`,
    for k=3 it calls `stream_batches_2layers_H3` implemented above.
    This keeps a uniform API: yields batches (X, Z, H, y, params) where Z is flattened Hk.
    """
    if k == 2:
        # stream_batches_teacher yields (X, H, y, A_teacher, B) where H is h (bs,p)
        # to keep a uniform (X,Z,H,y,params) we wrap it to also return Z
        def _factory():
            def _stream():
                for X, H, y, A_teacher, B in stream_batches_teacher(d=d, p=p, n=n, batch_size=batch_size, A_mode=A_mode, beta=beta, seed=seed, device=device, input_mode=input_mode, gamma=gamma):
                    Z = flatten_H2_of_X_sym_hermite(X)
                    params = {"A": A_teacher, "B": B, "mean_y": None, "std_y": None}
                    yield X, Z, H, y, params
            return _stream
        return _factory()

    elif k == 3:
        # our stream_batches_2layers_H3 already matches the desired signature
        return stream_batches_2layers_H3(d=d, p=p, n=n, batch_size=batch_size, beta=beta, seed=seed, device=device, normalize_y=normalize_y, return_params=return_params, gamma=gamma)

    else:
        raise ValueError(f"stream_batches_teacher_k only supports k=2 or k=3 (got k={k})")


@torch.no_grad()
def gen_Aflat_orth(p: int, m: int, device=None, seed: int = 0):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    M = torch.randn(m, p, generator=g, device=device, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)     # (m,p) orthonormal columns
    A = Q.T.contiguous()          # (p,m) orthonormal rows
    return A.to(dtype=torch.float32)


@torch.no_grad()
def stream_batches_2layers_H3(
    d: int, p: int, n: int, batch_size: int,
    beta: float,
    seed: int,
    device,
    normalize_y: bool = True,
    return_params: bool = True,
    g_name: str = None,
    g_callable=None,
    gamma: float = DEFAULT_GAMMA,
):
    m = d * (d + 1) * (d + 2) // 6

    # fixed teacher params for this run
    A = gen_Aflat_orth(p=p, m=m, device=device, seed=seed + 11)
    # B symmetric dense
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 23)
    M = torch.randn(p, p, generator=g, device=device, dtype=torch.float32)
    B = (M + M.T) / 2.0
    B = beta * B / math.sqrt(p)

    # estimate mean/std if asked
    mean_y = torch.tensor(0.0, device=device)
    std_y = torch.tensor(1.0, device=device)

    act = get_activation_fn(g_name=g_name, g_callable=g_callable)

    if normalize_y:
        count = 0
        mean = torch.tensor(0.0, device=device, dtype=torch.float64)
        M2 = torch.tensor(0.0, device=device, dtype=torch.float64)
        for _ in range((n + batch_size - 1) // batch_size):
            bs = min(batch_size, n - count)
            if bs <= 0:
                break
            X = torch.randn(bs, d, device=device, dtype=torch.float32)
            Z = flatten_H3_sym(X)
            H = Z @ A.T
            # compute scalar pre-activation s = h^T B h - Tr(B) and apply activation g on scalar
            s = (H @ B * H).sum(dim=1) - torch.trace(B)
            y = act(s)
            y64 = y.to(torch.float64)

            # Welford
            count_new = count + bs
            batch_mean = y64.mean()
            batch_M2 = ((y64 - batch_mean) ** 2).sum()
            delta = batch_mean - mean
            mean = mean + delta * (bs / count_new)
            M2 = M2 + batch_M2 + delta**2 * (count * bs / count_new)
            count = count_new

        mean_y = mean.to(torch.float32)
        std_y = torch.sqrt((M2 / max(count - 1, 1))).to(torch.float32) + 1e-12

    def stream_fn():
        seen = 0
        while seen < n:
            bs = min(batch_size, n - seen)
            X = torch.randn(bs, d, device=device, dtype=torch.float32)
            Z = flatten_H3_sym(X)
            H = Z @ A.T
            # compute scalar pre-activation s = h^T B h - Tr(B) and apply activation g on scalar
            s = (H @ B * H).sum(dim=1) - torch.trace(B)
            y = act(s)
            y = (y - mean_y) / std_y if normalize_y else y
            seen += bs
            if return_params:
                yield X, Z, H, y, {
                    "A": A,
                    "B": B,
                    "mean_y": mean_y,
                    "std_y": std_y,
                    "m": m,
                }
            else:
                yield X, Z, H, y, None

    return stream_fn

@torch.no_grad()
def hermite2_features(Z: torch.Tensor) -> torch.Tensor:
    """
    Vector Hermite-2 feature map, componentwise:
    H2(z)_j = z_j^2 - 1.
    Z: (bs, m)
    returns: (bs, m)
    """
    return Z * Z - 1.0



#########################################
############ Bacth 2 layers #############
#########################################

def stream_batches_teacher(d, p, n, batch_size, A_mode=None, beta=None, seed=0, device=None, g_name=None, g_callable=None, input_mode='true', B_mode="dense", gamma=DEFAULT_GAMMA):
    """
    Yields batches (X, H, y) on GPU, without storing full dataset.
    Also returns (A_teacher, B) for reference.
    """
    if A_mode is None:
        A_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # teacher params
    if A_mode == "rank1_orth":
        A_teacher = gen_A_rank1_orth_torch(d, p, gen, device)
    elif A_mode == "sym_orth_frob":
        A_teacher = gen_A_sym_orth_frob_torch(d, p, gen, device)
    else:
        raise ValueError("A_mode must be 'rank1_orth' or 'sym_orth_frob'")

    if B_mode == "dense":
        B = gen_B_symmetric_dense_torch(p, gen, device, beta=beta)
    elif B_mode == "powerlaw_diag":
        B = gen_B_powerlaw_diag_torch(p, gen, device, beta=beta, gamma=gamma, rademacher=True)
    else:
        raise ValueError(f"Unknown B_mode: {B_mode}")


    Aflat = flatten_A_sym_for_H2_feature(A_teacher["A"]).to(device)
    m = Aflat.shape[1]

    # stream
    seen = 0
    while seen < n:
        bs = min(batch_size, n - seen)
        if input_mode == 'gauss':
            # gaussian-equivalent: sample Hermite features Z ~ N(0,I_m) and map via Aflat
            if A_mode != 'sym_orth_frob':
                raise ValueError("input_mode='gauss' currently only supported for A_mode='sym_orth_frob'")
            Z = torch.randn(bs, m, generator=gen, device=device)
            H = Z @ Aflat.T
            X_or_Z = Z
        else:
            X = torch.randn(bs, d, generator=gen, device=device)
            H = compute_h_from_X_torch(X, A_teacher)
            X_or_Z = X
        # compute scalar pre-activation s = h^T B h - Tr(B), then apply activation g on scalar
        s = compute_y_from_H_torch(H, B)
        act = get_activation_fn(g_name=g_name, g_callable=g_callable)
        y = act(s)

        yield X_or_Z, H, y, A_teacher, B
        seen += bs

@torch.no_grad()
def compute_mean_std_y_stream(d, p, n, batch_size, A_mode=None, beta=None, seed=0, device=None, g_name=None, g_callable=None, input_mode='true', B_mode="dense", gamma=DEFAULT_GAMMA):
    """
    1 pass streaming on the teacher to estimate mean/std of y on the dataset.
    Return mean_y, std_y (scalars torch float64 on CPU).
    """
    if A_mode is None:
        A_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    count = 0
    mean = torch.tensor(0.0, device=device, dtype=torch.float64)
    M2 = torch.tensor(0.0, device=device, dtype=torch.float64)

    for X, H, y, A_teacher, B in stream_batches_teacher(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode=A_mode, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=g_callable,
        input_mode=input_mode, 
        B_mode=B_mode, gamma=gamma,
    ):
        y64 = y.to(torch.float64)
        bs = y64.numel()
        count_new = count + bs

        # batch mean/M2
        batch_mean = y64.mean()
        batch_M2 = ((y64 - batch_mean) ** 2).sum()

        # merge online (Welford merge)
        delta = batch_mean - mean
        mean = mean + delta * (bs / count_new) if count_new > 0 else batch_mean
        M2 = M2 + batch_M2 + delta**2 * (count * bs / count_new) if count > 0 else batch_M2

        count = count_new

    var = M2 / max(count, 1)
    std = torch.sqrt(var + 1e-12)

    return mean.detach().cpu(), std.detach().cpu()


def stream_batches_teacher_y_normalized(d, p, n, batch_size, A_mode=None, beta=None, seed=0, device=None, mean_y=None, std_y=None, g_name=None, g_callable=None, input_mode='true', B_mode='dense', gamma=DEFAULT_GAMMA):
    """
    Replay the teacher (same seed), but gives y_norm = (y - mean_y)/std_y.
    mean_y, std_y should come from compute_mean_std_y_stream (CPU scalars).
    """
    if A_mode is None:
        A_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    if mean_y is None or std_y is None:
        mean_y, std_y = compute_mean_std_y_stream(d=d, p=p, n=n, batch_size=batch_size, A_mode=A_mode, beta=beta, seed=seed, device=device, input_mode=input_mode, g_name=g_name, g_callable=g_callable, B_mode=B_mode, gamma=gamma)
    mean_y = mean_y.to(device=device, dtype=torch.float32)
    std_y = std_y.to(device=device, dtype=torch.float32)

    for X, H, y, A_teacher, B in stream_batches_teacher(
        d=d, p=p, n=n, batch_size=batch_size,
        A_mode=A_mode, beta=beta, seed=seed, device=device,
        g_name=g_name, g_callable=g_callable,
        input_mode=input_mode, 
        B_mode=B_mode, gamma=gamma,
    ):
        y_norm = (y - mean_y) / std_y
        yield X, H, y_norm, A_teacher, B



#########################################
########## Patches for 3 layers #########
#########################################


def compute_h2_from_H1_torch(H1, A2_teacher):
    # H1: (bs, p1)
    mode = A2_teacher["mode"]

    if mode == "sym_orth_frob":
        A2 = A2_teacher["A"]  # (p2,p1,p1)
        # Compute empirical first/second moments of H1 on the provided batch
        bs = max(H1.shape[0], 1)
        mean_h1 = H1.mean(dim=0)                     # (p1,)
        H1c = H1 - mean_h1[None, :]
        # Empirical second moment E[h h^T] = Cov + mean mean^T
        E = (H1c.t() @ H1c) / bs + torch.ger(mean_h1, mean_h1)

        # Quadratic form per feature
        quad = torch.einsum("bp,qpr,br->bq", H1, A2, H1)      # (bs,p2)

        # Mean term is tr(A_i * E) for each i
        mean_term = torch.einsum("qij,ij->q", A2, E)         # (p2,)

        # Centered pre-activation s = h^T A h - E[h^T A h]
        s = quad - mean_term[None, :]

        # Normalize variance per coordinate so each feature has unit variance.
        # Under Gaussian assumption Var(h^T A h) = 2 * tr((A E)^2).
        A2E = torch.einsum("qij,jk->qik", A2, E)             # (p2,p1,p1)
        fro2 = torch.einsum("qik,qik->q", A2E, A2E)          # (p2,)
        var_s = 2.0 * fro2
        denom = torch.sqrt(var_s + 1e-12)

        H2 = s / denom[None, :]
        return H2

    elif mode == "rank1_orth":
        # A2_teacher["U"]: (p1, p2), columns orthonormal
        U = A2_teacher["U"]
        Z = H1 @ U                      # (bs, p2)
        # Center the squared feature by its expectation and normalize its variance
        Ez2 = (Z**2).mean(dim=0)                            # (p2,)
        s = Z**2 - Ez2[None, :]
        var_s = ((s)**2).mean(dim=0)
        denom = torch.sqrt(var_s + 1e-12)
        H2 = s / denom[None, :]
        return H2

    else:
        raise ValueError("Unknown A2_teacher mode")


def stream_batches_teacher_3layer(d, p1, p2, n, batch_size, A1_mode=None, A2_mode=None, beta=None, seed=0, device=None, gamma=DEFAULT_GAMMA):
    """
    Yields batches (X, H1, H2, y) on GPU, without storing full dataset.
    Also returns (A1_teacher, A2_teacher, B) for reference.
    """
    if A1_mode is None:
        A1_mode = DEFAULT_A_MODE
    if A2_mode is None:
        A2_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # teacher params
    # - A1
    if A1_mode == "rank1_orth":
        A1_teacher = gen_A_rank1_orth_torch(d, p1, gen, device)
    elif A1_mode == "sym_orth_frob":
        A1_teacher = gen_A_sym_orth_frob_torch(d, p1, gen, device)
    else:
        raise ValueError("A_mode must be 'rank1_orth' or 'sym_orth_frob'")
    # - A2
    if A2_mode == "rank1_orth":
        A2_teacher = gen_A_rank1_orth_torch(p1, p2, gen, device)
    elif A2_mode == "sym_orth_frob":
        A2_teacher = gen_A_sym_orth_frob_torch(p1, p2, gen, device)
    else:
        raise ValueError("A_mode must be 'rank1_orth' or 'sym_orth_frob'")
    # - B
    B = gen_B_symmetric_dense_torch(p2, gen, device, beta=beta)

    # stream
    seen = 0
    while seen < n:
        bs = min(batch_size, n - seen)
        X = torch.randn(bs, d, generator=gen, device=device)
        H1 = compute_h_from_X_torch(X, A1_teacher, gamma=gamma)
        H2 = compute_h2_from_H1_torch(H1, A2_teacher)
        y = compute_y_from_H_torch(H2, B)
        yield X, H1, H2, y, A1_teacher, A2_teacher, B
        seen += bs

def compute_mean_std_y_stream_3layers(d, p1, p2, n, batch_size, A1_mode=None, A2_mode=None, beta=None, seed=0, device=None, gamma=DEFAULT_GAMMA):
    """
    1 pass streaming on the teacher to estimate mean/std of y on the dataset.
    Return mean_y, std_y (scalars torch float64 on CPU).
    """
    if A1_mode is None:
        A1_mode = DEFAULT_A_MODE
    if A2_mode is None:
        A2_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    count = 0
    mean = torch.tensor(0.0, device=device, dtype=torch.float64)
    M2 = torch.tensor(0.0, device=device, dtype=torch.float64)

    for X, H1, H2, y, A1_teacher, A2_teacher, B in stream_batches_teacher_3layer(
        d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
        A1_mode=A1_mode, A2_mode=A2_mode, beta=beta, seed=seed, device=device,
        gamma=gamma
    ):
        y64 = y.to(torch.float64)
        bs = y64.numel()
        count_new = count + bs

        # batch mean/M2
        batch_mean = y64.mean()
        batch_M2 = ((y64 - batch_mean) ** 2).sum()

        # merge online (Welford merge)
        delta = batch_mean - mean
        mean = mean + delta * (bs / count_new) if count_new > 0 else batch_mean
        M2 = M2 + batch_M2 + delta**2 * (count * bs / count_new) if count > 0 else batch_M2

        count = count_new

    var = M2 / max(count, 1)
    std = torch.sqrt(var + 1e-12)

    return mean.detach().cpu(), std.detach().cpu()

def stream_batches_teacher_3layers_y_normalized(d, p1, p2, n, batch_size, A1_mode=None, A2_mode=None, beta=None, seed=0, device=None, mean_y=None, std_y=None, gamma=DEFAULT_GAMMA):
    """
    Replay the teacher (same seed), but gives y_norm = (y - mean_y)/std_y.
    mean_y, std_y should come from compute_mean_std_y_stream (CPU scalars).
    """
    if A1_mode is None:
        A1_mode = DEFAULT_A_MODE
    if A2_mode is None:
        A2_mode = DEFAULT_A_MODE
    if beta is None:
        beta = DEFAULT_BETA
    if device is None:
        device = get_device()
    if mean_y is None or std_y is None:
        mean_y, std_y = compute_mean_std_y_stream_3layers(d=d, p1=p1, p2=p2, n=n, batch_size=batch_size, A1_mode=A1_mode, A2_mode=A2_mode, beta=beta, seed=seed, device=device, gamma=gamma)
    mean_y = mean_y.to(device=device, dtype=torch.float32)
    std_y = std_y.to(device=device, dtype=torch.float32)

    for X, H1, H2, y, A1_teacher, A2_teacher, B in stream_batches_teacher_3layer(
        d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
        A1_mode=A1_mode, A2_mode=A2_mode, beta=beta, seed=seed, device=device,
        gamma=gamma
    ):
        y_norm = (y - mean_y) / std_y
        yield X, H1, H2, y_norm, A1_teacher, A2_teacher, B


#------------------------------------------#
# Adding skip connections for the 3 layers #
#------------------------------------------#

def gen_b_gaussian_torch(p, gen, device, scale=1.0):
    b = torch.randn(p, generator=gen, device=device)
    b = b / (torch.linalg.norm(b) + 1e-12)
    return scale * b

@torch.no_grad()
def stream_batches_hier2_skip(
    d, p1, p2, n, batch_size,
    A_mode_1="sym_orth_frob",
    A_mode_2="sym_orth_frob",
    b_scale_1=1.0,
    b_scale_2=1.0,
    seed=0,
    device=None,
    apply_tau=False,
):
    """
    Streaming teacher for hierarchical L=2 with skip:
      x ~ N(0,I_d)
      h1_i = <A1_i, H2(x)> / sqrt(2)
      h2_j = <A2_j, H2(h1)> / sqrt(2)
      y = b1·h1 + b2·h2
    Optionally apply tau(y)= y/(1+|y|) if apply_tau=True.

    Yields (X, y, params_dict) where:
      X: (bs,d), y:(bs,)
      params_dict contains teacher params (A1, A2, b1, b2)
    """
    if device is None:
        device = get_device()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # A1
    if A_mode_1 == "rank1_orth":
        A1 = gen_A_rank1_orth_torch(d, p1, gen, device)
    elif A_mode_1 == "sym_orth_frob":
        A1 = gen_A_sym_orth_frob_torch(d, p1, gen, device)
    else:
        raise ValueError("A_mode_1 must be 'rank1_orth' or 'sym_orth_frob'")

    # A2 acts on h1-space => dimension p1
    if A_mode_2 == "rank1_orth":
        A2 = gen_A_rank1_orth_torch(p1, p2, gen, device)
    elif A_mode_2 == "sym_orth_frob":
        A2 = gen_A_sym_orth_frob_torch(p1, p2, gen, device)
    else:
        raise ValueError("A_mode_2 must be 'rank1_orth' or 'sym_orth_frob'")

    b1 = gen_b_gaussian_torch(p1, gen, device, scale=b_scale_1)
    b2 = gen_b_gaussian_torch(p2, gen, device, scale=b_scale_2)

    params = {"A1": A1, "A2": A2, "b1": b1, "b2": b2}

    seen = 0
    while seen < n:
        bs = min(batch_size, n - seen)
        X = torch.randn(bs, d, generator=gen, device=device)

        h1 = compute_h_from_X_torch(X, A1, gamma=None)          # (bs,p1)
        h2 = compute_h_from_X_torch(h1, A2)         # (bs,p2)

        y = (h1 @ b1) + (h2 @ b2)                   # (bs,)
        if apply_tau:
            y = y / (1.0 + torch.abs(y))

        yield X, y, params
        seen += bs


@torch.no_grad()
def compute_mean_std_y_stream_hier2_skip(
    d, p1, p2, n, batch_size,
    A_mode_1, A_mode_2,
    b_scale_1, b_scale_2,
    seed, device,
    apply_tau=False
):
    """
    1 pass streaming to estimate mean/std of y for normalization.
    """
    count = 0
    s1 = torch.tensor(0.0, device=device, dtype=torch.float64)
    s2 = torch.tensor(0.0, device=device, dtype=torch.float64)

    stream = stream_batches_hier2_skip(
        d=d, p1=p1, p2=p2, n=n, batch_size=batch_size,
        A_mode_1=A_mode_1, A_mode_2=A_mode_2,
        b_scale_1=b_scale_1, b_scale_2=b_scale_2,
        seed=seed, device=device, apply_tau=apply_tau
    )
    for X, y, _ in stream:
        y64 = y.to(torch.float64)
        count += y64.numel()
        s1 += y64.sum()
        s2 += (y64 * y64).sum()

    mean = (s1 / max(count, 1)).cpu()
    var = (s2 / max(count, 1) - mean.to(torch.float64) ** 2).clamp_min(1e-18).cpu()
    std = torch.sqrt(var)
    return mean, std



