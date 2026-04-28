#!/usr/bin/env python3
"""Create tasks.jsonl for 2-layer experiments with *chunking over alphas*.

Supports:
  --alphas "1.0:3.8:0.05"  (start:stop:step, inclusive)
  --alphas "1.0,1.05,..."  (CSV)

Also supports disabling early stop:
  --stop_tol none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_range(s: str) -> list[int]:
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    return [int(s)]


def _parse_alphas(s: str) -> list[float]:
    s = s.strip()
    # range form: "start:stop:step"
    if ":" in s:
        a, b, h = s.split(":")
        a, b, h = float(a), float(b), float(h)
        if h <= 0:
            raise ValueError("alpha step must be > 0")
        # inclusive range with rounding to avoid float drift
        out = []
        k = 0
        while True:
            val = a + k * h
            if val > b + 1e-12:
                break
            out.append(round(val, 10))
            k += 1
        return out

    # CSV form
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_stop_tol(s: str):
    s = str(s).strip().lower()
    if s in {"none", "null", "no", "off"}:
        return None
    return float(s)


def chunk_list(xs: list[float], chunk_size: int) -> list[list[float]]:
    return [xs[i: i + chunk_size] for i in range(0, len(xs), chunk_size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--exp_id", type=str, default="2L")
    ap.add_argument("--d", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--alphas", type=str, required=True, help='CSV "1.0,1.1" or range "1.0:3.8:0.05"')
    ap.add_argument("--chunk_size", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0-9", help="e.g. 0-9 or 0,1,2")
    ap.add_argument("--models", type=str, default="true,gauss")

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_test", type=int, default=5000)

    ap.add_argument("--A_mode_teacher", type=str, default="sym_orth_frob")
    ap.add_argument("--B_mode", type=str, default="powerlaw_diag")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--g_name", type=str, default="id")
    ap.add_argument("--save_estimates", action="store_true", help="Save Ahat/Bhat per alpha for posthoc analysis")

    ap.add_argument("--n_iter_C_max", type=int, default=15)
    ap.add_argument("--oversamp_C", type=int, default=10)
    ap.add_argument("--T_min", type=int, default=0)
    ap.add_argument("--stop_tol", type=str, default="none", help='e.g. "1e-2" or "none"')

    ap.add_argument("--fit_degree", type=int, default=5)
    ap.add_argument("--fit_ridge", type=float, default=1e-6)
    ap.add_argument(
        "--head_mode",
        type=str,
        default="spectral_B",
        choices=["spectral_B", "latent_rbf", "input_rbf", "latent_poly2", "input_poly4_rf", "latent_rf_spectral"],
    )
    ap.add_argument("--n_krr_max", type=int, default=4000)
    ap.add_argument("--rbf_lambda", type=float, default=1e-4)
    ap.add_argument("--rbf_sigma_mult", type=float, default=1.0)
    ap.add_argument("--rbf_standardize", action="store_true", default=True)
    ap.add_argument("--no_rbf_standardize", dest="rbf_standardize", action="store_false")
    ap.add_argument("--poly_lambda", type=float, default=1e-4)
    ap.add_argument("--m_rf", type=int, default=1024)
    ap.add_argument("--layer1_mode", type=str, default="hermite_spectral",
                choices=["hermite_spectral", "rf_spectral"])
    ap.add_argument("--rf_width", type=int, default=8192)
    ap.add_argument("--rf_activation", type=str, default="relu")
    ap.add_argument("--rf2_width", type=int, default=4096)
    ap.add_argument("--rf2_activation", type=str, default="relu_raw")
    ap.add_argument("--rf2_affine_ridge", type=float, default=1e-6)
    ap.add_argument("--rf2_use_whiten", action="store_true")
    ap.add_argument("--no_rf2_use_whiten", dest="rf2_use_whiten", action="store_false")
    ap.set_defaults(rf2_use_whiten=True)

    ap.add_argument("--calibrate_output", action="store_true")
    ap.add_argument("--no_calibrate_output", dest="calibrate_output", action="store_false")
    ap.set_defaults(calibrate_output=False)

    ap.add_argument("--load_ahat_exp_id", type=str, default="")

    args = ap.parse_args()

    alphas = sorted(_parse_alphas(args.alphas))
    seeds = _parse_range(args.seeds)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stop_tol = _parse_stop_tol(args.stop_tol)

    chunks = chunk_list(alphas, int(args.chunk_size))

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    chunk_id = 0
    for seed in seeds:
        for chunk in chunks:
            task = {
                "exp_id": args.exp_id,
                "chunk_id": chunk_id,
                "d": int(args.d),
                "eps": float(args.eps),
                "alphas": [float(a) for a in chunk],
                "seed": int(seed),
                "beta": float(args.beta),
                "batch_size": int(args.batch_size),
                "n_test": int(args.n_test),
                "A_mode_teacher": args.A_mode_teacher,
                "B_mode": args.B_mode,
                "gamma": float(args.gamma),
                "save_estimates": bool(args.save_estimates),
                "models": models,
                "g_name": args.g_name,
                "n_iter_C_max": int(args.n_iter_C_max),
                "oversamp_C": int(args.oversamp_C),
                "T_min": int(args.T_min),
                "stop_tol": stop_tol,
                "fit_degree": int(args.fit_degree),
                "fit_ridge": float(args.fit_ridge),
                "head_mode": str(args.head_mode),
                "layer1_mode": str(args.layer1_mode),
                "rf_width": int(args.rf_width),
                "rf_activation": str(args.rf_activation),
                "rf2_width": int(args.rf2_width),
                "rf2_activation": str(args.rf2_activation),
                "rf2_affine_ridge": float(args.rf2_affine_ridge),
                "n_krr_max": int(args.n_krr_max),
                "rbf_lambda": float(args.rbf_lambda),
                "rbf_sigma_mult": float(args.rbf_sigma_mult),
                "rbf_standardize": bool(args.rbf_standardize),
                "poly_lambda": float(args.poly_lambda),
                "m_rf": int(args.m_rf),
                "calibrate_output": bool(args.calibrate_output),
                "load_ahat_exp_id": None if args.load_ahat_exp_id == "" else str(args.load_ahat_exp_id),
            }
            lines.append(json.dumps(task))
            chunk_id += 1

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} tasks to {outpath}")


if __name__ == "__main__":
    main()