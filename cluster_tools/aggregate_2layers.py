#!/usr/bin/env python3
"""Aggregate metrics.jsonl files into a clean summary table (mean/std over seeds)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def iter_metrics_files(root: Path):
    for p in root.rglob("metrics.jsonl"):
        yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="results directory (root)")
    ap.add_argument("--outdir", type=str, required=True, help="output directory")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fpath in iter_metrics_files(indir):
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    if not rows:
        raise SystemExit(f"No metrics found under {indir}")

    df = pd.DataFrame(rows)

    # Columns to aggregate if present
    metric_cols = [c for c in ["nmse", "mse", "baseline", "ovA", "ovH", "corr_s", "eig_err_B", "eig_corr_B", "wall_seconds"] if c in df.columns]

    # Group keys (keep this stable for paper plots)
    group_cols = [c for c in [
        "exp_id", "d", "p", "eps", "alpha", "n", "model",
        "A_mode_teacher", "B_mode", "gamma", "g_name",
        "batch_size", "n_iter_C_max", "oversamp_C"
    ] if c in df.columns]

    # If exp_id isn't in metrics, infer nothing (still ok)
    if "exp_id" not in group_cols:
        # If you want, you can inject exp_id via directory name later
        pass

    agg = {}
    for c in metric_cols:
        agg[c] = ["mean", "std", "count"]

    g = df.groupby(group_cols, dropna=False).agg(agg).reset_index()

    # Flatten MultiIndex columns
    g.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in g.columns]

    # Nice rename
    rename_map = {}
    for c in metric_cols:
        rename_map[f"{c}_mean"] = f"{c}_mean"
        rename_map[f"{c}_std"] = f"{c}_std"
        rename_map[f"{c}_count"] = f"{c}_count"
    g = g.rename(columns=rename_map)

    out_csv = outdir / "summary.csv"
    g.to_csv(out_csv, index=False)

    # Also save raw for debugging
    df.to_parquet(outdir / "raw_metrics.parquet", index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {outdir / 'raw_metrics.parquet'}")


if __name__ == "__main__":
    main()