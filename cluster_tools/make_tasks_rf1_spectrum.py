#!/usr/bin/env python3
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
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _parse_alphas(s: str) -> list[float]:
    s = s.strip()
    if ":" in s:
        a, b, h = s.split(":")
        a, b, h = float(a), float(b), float(h)
        out = []
        k = 0
        while True:
            val = a + k * h
            if val > b + 1e-12:
                break
            out.append(round(val, 10))
            k += 1
        return out
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_stop_tol(s: str):
    s = str(s).strip().lower()
    if s in {"none", "null", "no", "off"}:
        return None
    return float(s)


def chunk_list(xs: list[float], chunk_size: int) -> list[list[float]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--exp_id", type=str, default="rf1_spectrum")
    ap.add_argument("--d", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--alphas", type=str, required=True)
    ap.add_argument("--chunk_size", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="0-9")

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_test", type=int, default=3000)

    ap.add_argument("--A_mode_teacher", type=str, default="sym_orth_frob")
    ap.add_argument("--B_mode", type=str, default="dense")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--g_name", type=str, default="id")

    ap.add_argument("--n_iter_C_max", type=int, default=15)
    ap.add_argument("--oversamp_C", type=int, default=10)
    ap.add_argument("--T_min", type=int, default=0)
    ap.add_argument("--stop_tol", type=str, default="none")

    ap.add_argument("--rf_width", type=int, default=8192)
    ap.add_argument("--rf_activation", type=str, default="relu_l1")

    args = ap.parse_args()

    alphas = sorted(_parse_alphas(args.alphas))
    seeds = _parse_range(args.seeds)
    stop_tol = _parse_stop_tol(args.stop_tol)
    chunks = chunk_list(alphas, int(args.chunk_size))

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    chunk_id = 0
    for seed in seeds:
        for chunk in chunks:
            task = {
                "exp_id": str(args.exp_id),
                "chunk_id": chunk_id,
                "d": int(args.d),
                "eps": float(args.eps),
                "alphas": [float(a) for a in chunk],
                "seed": int(seed),
                "beta": float(args.beta),
                "batch_size": int(args.batch_size),
                "n_test": int(args.n_test),
                "A_mode_teacher": str(args.A_mode_teacher),
                "B_mode": str(args.B_mode),
                "gamma": float(args.gamma),
                "g_name": str(args.g_name),
                "n_iter_C_max": int(args.n_iter_C_max),
                "oversamp_C": int(args.oversamp_C),
                "T_min": int(args.T_min),
                "stop_tol": stop_tol,
                "rf_width": int(args.rf_width),
                "rf_activation": str(args.rf_activation),
            }
            lines.append(json.dumps(task))
            chunk_id += 1

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} tasks to {outpath}")


if __name__ == "__main__":
    main()