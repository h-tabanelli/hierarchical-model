# Code for "Deep Learning of Compositional Targets with Hierarchical Spectral Methods"

Anonymous submission — NeurIPS 2026.

## Overview

This repository contains self-contained Python code for the experiments in the paper.
The two main scripts produce the data underlying the figures:

| Script | Produces |
|--------|----------|
| `run_2layers.py` | NMSE vs. α sweep (Figures 2, 3, …) |
| `run_rf1_spectrum.py` | Eigenspectrum of C in the RF basis (Figure on RF1 spectrum) |

The core library modules are:

| Module | Content |
|--------|---------|
| `teacher.py` | Teacher model: generates A, B, streaming (X, y) batches |
| `estimators.py` | All spectral estimators (hermite, RF1, RF2, heads) |
| `measures.py` | Overlap metrics, eigenvalue comparisons, plotting utilities |

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for large d; all scripts fall back to CPU automatically.

## Quick start (CPU, small d)

```bash
# 2-layer NMSE sweep — ~2 min on CPU
python run_2layers.py \
    --d 50 --eps 0.5 \
    --alphas "1.5:3.5:0.5" \
    --seeds 0,1 \
    --models true,gauss \
    --head_mode spectral_B \
    --B_mode powerlaw_diag --gamma 0.25 \
    --outdir results/demo_2layers

# RF1 spectrum — ~1 min on CPU (small rf_width for speed)
python run_rf1_spectrum.py \
    --d 50 --eps 0.5 \
    --alphas "1.5:3.5:0.5" \
    --seeds 0,1 \
    --rf_width 512 \
    --outdir results/demo_rf1
```

## Reproducing paper figures (GPU recommended)

```bash
# Main NMSE figure (d=100, dense alpha grid, 5 seeds) — ~4 h on a single GPU
python run_2layers.py \
    --d 100 --eps 0.5 \
    --alphas "1.0:4.0:0.1" \
    --seeds 0-4 \
    --models true,gauss \
    --head_mode spectral_B \
    --B_mode powerlaw_diag --gamma 0.25 \
    --n_iter_C_max 15 --oversamp_C 10 \
    --outdir results/d100_spectralB

# RF1 spectrum figure (d=100)
python run_rf1_spectrum.py \
    --d 100 --eps 0.5 \
    --alphas "1.0:4.0:0.25" \
    --seeds 0-4 \
    --rf_width 4096 \
    --outdir results/d100_rf1spec
```

## Output format

### `run_2layers.py`

```
results/<exp_id>/seed=XXXX/metrics.jsonl
```

Each line is a JSON record with fields including `alpha`, `n`, `d`, `p`, `model`, `nmse`,
`ovA` (subspace overlap with teacher A), `corr_s`, `eig_err_B`, and timing information.

### `run_rf1_spectrum.py`

```
results/<exp_id>/seed=XXXX/rf1_spectrum_alpha=X.XXXX.pt
results/<exp_id>/seed=XXXX/rf1_spectrum_metrics.jsonl
```

Each `.pt` file contains a dict with `full_eigs` (full eigenspectrum of C),
`selected_ritz_eigs` (Ritz values on the learned subspace), and per-neuron statistics.

## Key parameters

### Model

| Parameter | Meaning |
|-----------|---------|
| `--d` | Input dimension |
| `--eps` | Hidden dimension exponent: p = ⌊d^eps⌋ |
| `--alphas` | Sample exponent: n = ⌊d^alpha⌋. Format: `"start:stop:step"` or CSV |
| `--seeds` | RNG seeds: `"0-4"` or `"0,1,2"` |
| `--models` | `true` (Gaussian X) and/or `gauss` (Gaussian-equivalent) |
| `--B_mode` | `dense` or `powerlaw_diag` |
| `--gamma` | Power-law decay exponent for B eigenvalues |
| `--g_name` | Target nonlinearity: `id`, `relu`, `erf`, … |

### Estimator

| Parameter | Meaning |
|-----------|---------|
| `--head_mode` | Second-layer head: `spectral_B` (main), `latent_rbf`, `latent_rf_spectral`, … |
| `--layer1_mode` | First-layer method: `hermite_spectral` (main) or `rf_spectral` |
| `--n_iter_C_max` | Randomized power-iteration passes (default 15) |
| `--oversamp_C` | Oversampling factor for randomized SVD (default 10) |
