"""
SLQ-based spectral density estimation support has been removed from this repo.

If you previously used `run_spectrum_slq.py`, please migrate to `run_2layers_spectrum.py`
which uses `dense_spectrum_from_stream` to compute full eigenvalues of moment matrices
and is compatible with the plotting used in the paper.

This file now only provides a helpful message for users.
"""

def run_spectrum_Ctilde_slq(*args, **kwargs):
    raise RuntimeError(
        "SLQ-based spectral density estimation was removed. Use run_2layers_spectrum.py or estimators.dense_spectrum_from_stream instead."
    )
