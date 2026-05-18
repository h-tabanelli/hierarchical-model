#!/usr/bin/env python3
"""
Compute the full eigenspectrum of the true 2-layer power-law model.

Builds the L1 (input representation, dim m=d(d+1)/2) and L2 (hidden layer, dim p=d^p_exp)
covariance matrices from data streams and diagonalises them exactly, yielding bulk + spikes.

Output is compatible with the panel-C style plot in plots/plot_paper_abc.py.

Typical runtimes on RTX 6000 Ada (50 GB VRAM):
  d=100  m=5050   alpha=3  ~  1 min / seed
  d=140  m=9870   alpha=3  ~  4 min / seed   (m ≈ RF rf_width=10000, nice comparison)
  d=200  m=20100  alpha=3  ~ 15 min / seed
  d=300  m=45150  alpha=3  ~ 60 min / seed   (eigvalsh dominates)

Usage:
  python run_2layers_spec_powerlaw.py --d 140 --alpha 3.0 --gamma 0.4 --g_name tanh --seeds 0 1 2 3 4

The per-seed npz files and a merged eigs_all_seeds.npz are written to
  results/2layers_spectrum_d{D}_a{alpha}_g{gamma}_{g_name}/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_2layers_spectrum import run_2layers_spectra_per_alpha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d",          type=int,   default=140,  help="Input dimension")
    ap.add_argument("--alpha",      type=float, default=3.0,  help="n = round(d^alpha) samples")
    ap.add_argument("--p_exp",      type=float, default=0.5,  help="p = round(d^p_exp) hidden dim")
    ap.add_argument("--gamma",      type=float, default=0.4,  help="Power-law exponent for B")
    ap.add_argument("--g_name",     type=str,   default="tanh", choices=["id", "tanh", "relu_centered"])
    ap.add_argument("--seeds",      type=int,   nargs="+", default=list(range(5)))
    ap.add_argument("--batch_size", type=int,   default=2048)
    ap.add_argument("--out_dir",    type=str,   default=None)
    ap.add_argument("--do_tau",     action="store_true", help="Also compute tau-wrapped spectra")
    ap.add_argument("--do_ge",      action="store_true", help="Also compute Gaussian-equivalent spectra (off by default)")
    ap.add_argument("--l1_only",    action="store_true", help="Skip L2 spectrum (saves time for large d)")
    args = ap.parse_args()

    d       = args.d
    p       = max(1, round(d ** args.p_exp))
    n       = round(d ** args.alpha)
    m       = d * (d + 1) // 2
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag     = f"d{d}_a{args.alpha:.1f}_g{args.gamma:.2f}_{args.g_name}"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"2layers_spectrum_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  d={d}  p={p}  n={n:,}  m(L1)={m:,}")
    print(f"  gamma={args.gamma}  g_name={args.g_name}  alpha={args.alpha}")
    print(f"  seeds={args.seeds}  device={device}")
    print(f"  out_dir={out_dir}")
    print(f"  L1 matrix memory (float64): {m**2 * 8 / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    all_eigs: dict[str, list[np.ndarray]] = {}
    t_total = time.time()

    for seed in args.seeds:
        print(f"--- Seed {seed} ---")
        t0 = time.time()
        seed_dir = out_dir / f"seed={seed:04d}"

        specs = run_2layers_spectra_per_alpha(
            d=d,
            p=p,
            n=n,
            batch_size=args.batch_size,
            A_mode="sym_orth_frob",
            beta=1.0,
            seed=seed,
            device=device,
            out_dir=seed_dir,
            do_tau=args.do_tau,
            do_sanity=False,
            do_ge=args.do_ge,
            g_name=args.g_name,
            gamma=args.gamma,
        )

        for name, evals in specs.items():
            all_eigs.setdefault(name, []).append(np.asarray(evals).reshape(-1))

        print(f"    done in {time.time() - t0:.1f}s")

    # Aggregate: shape (n_seeds, dim) for each spectrum
    agg = {name: np.stack(arrs, axis=0) for name, arrs in all_eigs.items()}
    agg_path = out_dir / "eigs_all_seeds.npz"
    np.savez_compressed(agg_path, **agg)

    # Save run metadata as a small JSON
    import json
    meta = dict(d=d, p=p, n=n, alpha=args.alpha, gamma=args.gamma,
                g_name=args.g_name, seeds=args.seeds,
                spectra=list(all_eigs.keys()))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nAll seeds done in {time.time() - t_total:.1f}s total")
    print(f"Saved: {agg_path}")
    for name, arr in agg.items():
        print(f"  {name}: {arr.shape}  ({arr.shape[1]} eigenvalues × {arr.shape[0]} seeds)")


if __name__ == "__main__":
    main()
