import torch
import math

# -----------------------------
# Sym indices i<=j<=k
# -----------------------------
_SYM3_CACHE = {}

def sym_triu_indices_3(d: int, device=None):
    key = (d, str(device))
    if key in _SYM3_CACHE:
        return _SYM3_CACHE[key]
    idx = torch.combinations(torch.arange(d, device=device), r=3, with_replacement=True)  # (m,3)
    _SYM3_CACHE[key] = idx
    """
    DEPRECATION SHIM
    -----------------
    This file used to contain H3-specific teacher utilities. The canonical
    implementations have been moved into `teacher.py`. Keeping this small shim
    preserves backwards compatibility for imports such as

        from teacher_h3 import flatten_H3_sym

    but emits a deprecation warning. You can safely remove `teacher_h3.py` once
    all callers import from `teacher` instead.
    """

    import warnings

    # re-export from the canonical module
    from teacher import (
        sym_triu_indices_3,
        flatten_H3_sym,
        gen_Aflat_orth,
        stream_batches_2layers_H3,
    )

    warnings.warn(
        "teacher_h3.py is deprecated — use functions from teacher.py instead",
        DeprecationWarning,
        stacklevel=2,
    )

    __all__ = [
        "sym_triu_indices_3",
        "flatten_H3_sym",
        "gen_Aflat_orth",
        "stream_batches_2layers_H3",
    ]