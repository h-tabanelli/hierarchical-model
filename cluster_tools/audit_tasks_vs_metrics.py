from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

EXPS = [
    "rf2cw_d120_tanh",
    "rf2cw_d140_tanh",
]

RUNS_ROOT = Path("runs")
RESULTS_ROOT = Path("results")
OUTDIR = Path("audit_runs")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# helpers
# ============================================================

def normalize_alphas(task: dict) -> list[float]:
    """
    Tries to recover the list of expected alphas from one task line.
    """
    for key in ["alphas", "alpha_list", "alpha_values"]:
        if key in task and task[key] is not None:
            vals = task[key]
            return [round(float(x), 10) for x in vals]

    for key in ["alpha", "alpha_start"]:
        if key in task and task[key] is not None:
            return [round(float(task[key]), 10)]

    raise KeyError(f"Could not find alpha field in task keys={list(task.keys())}")


def load_tasks(exp: str) -> list[dict]:
    candidates = [
        RUNS_ROOT / exp / "tasks.jsonl",
        RUNS_ROOT / f"{exp}.jsonl",
        RUNS_ROOT / f"tasks_{exp}.jsonl",
    ]

    taskfile = None
    for cand in candidates:
        if cand.exists():
            taskfile = cand
            break

    if taskfile is None:
        raise FileNotFoundError(
            f"Missing taskfile for {exp}. Tried:\n" + "\n".join(str(c) for c in candidates)
        )
    tasks = []
    with open(taskfile, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_task_id"] = i
            tasks.append(obj)
    return tasks


def collect_metrics_for_exp(exp: str) -> pd.DataFrame:
    rows = []
    for p in (RESULTS_ROOT / exp).rglob("metrics.jsonl"):
        chunk = None
        seed_from_path = None
        for part in p.parts:
            if part.startswith("chunk="):
                chunk = int(part.split("=")[1])
            if part.startswith("seed="):
                seed_from_path = int(part.split("=")[1])

        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["_metrics_path"] = str(p)
                obj["_chunk"] = chunk
                obj["_seed_path"] = seed_from_path
                rows.append(obj)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    if "alpha" in df.columns:
        df["alpha"] = df["alpha"].astype(float).round(10)
    return df


def build_expected_table(tasks: list[dict]) -> pd.DataFrame:
    rows = []
    for t in tasks:
        task_id = int(t["_task_id"])

        seed = t.get("seed", None)
        if seed is None:
            # fallback if some other key name is used
            seed = t.get("seed0", None)
        if seed is None:
            raise KeyError(f"Could not find seed in task {task_id}: keys={list(t.keys())}")

        seed = int(seed)
        alphas = normalize_alphas(t)

        for a in alphas:
            rows.append({
                "task_id": task_id,
                "seed": seed,
                "alpha": a,
            })
    return pd.DataFrame(rows)


def build_got_table(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return pd.DataFrame(columns=["task_id", "seed", "alpha"])

    rows = []
    for _, r in dfm.iterrows():
        task_id = r.get("_chunk", None)
        seed = r.get("seed", None)
        if pd.isna(seed):
            seed = r.get("_seed_path", None)
        alpha = r.get("alpha", None)

        if task_id is None or pd.isna(task_id):
            continue
        if seed is None or pd.isna(seed):
            continue
        if alpha is None or pd.isna(alpha):
            continue

        rows.append({
            "task_id": int(task_id),
            "seed": int(seed),
            "alpha": round(float(alpha), 10),
        })

    if not rows:
        return pd.DataFrame(columns=["task_id", "seed", "alpha"])

    got = pd.DataFrame(rows).drop_duplicates()
    return got


def audit_exp(exp: str):
    tasks = load_tasks(exp)
    expected = build_expected_table(tasks)
    metrics = collect_metrics_for_exp(exp)
    got = build_got_table(metrics)

    exp_out = OUTDIR / exp
    exp_out.mkdir(parents=True, exist_ok=True)

    expected.to_csv(exp_out / "expected.csv", index=False)
    got.to_csv(exp_out / "got.csv", index=False)

    if got.empty:
        missing = expected.copy()
    else:
        merged = expected.merge(
            got,
            on=["task_id", "seed", "alpha"],
            how="left",
            indicator=True,
        )
        missing = merged[merged["_merge"] == "left_only"][["task_id", "seed", "alpha"]].copy()

    missing.to_csv(exp_out / "missing.csv", index=False)

    # task-level summary
    exp_by_task = expected.groupby("task_id").size().rename("n_expected").reset_index()
    got_by_task = got.groupby("task_id").size().rename("n_got").reset_index() if not got.empty else pd.DataFrame(columns=["task_id", "n_got"])
    miss_by_task = missing.groupby("task_id").size().rename("n_missing").reset_index() if not missing.empty else pd.DataFrame(columns=["task_id", "n_missing"])

    summary = exp_by_task.merge(got_by_task, on="task_id", how="left").merge(miss_by_task, on="task_id", how="left")
    summary["n_got"] = summary["n_got"].fillna(0).astype(int)
    summary["n_missing"] = summary["n_missing"].fillna(0).astype(int)
    summary["is_complete"] = summary["n_missing"] == 0
    summary.to_csv(exp_out / "summary_by_task.csv", index=False)

    # compact json summary
    out = {
        "exp": exp,
        "n_tasks": int(expected["task_id"].nunique()),
        "n_expected_rows": int(len(expected)),
        "n_got_rows": int(len(got)),
        "n_missing_rows": int(len(missing)),
        "missing_task_ids": sorted(summary.loc[~summary["is_complete"], "task_id"].astype(int).tolist()),
    }
    with open(exp_out / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"===== {exp} =====")
    print(json.dumps(out, indent=2))


def main():
    for exp in EXPS:
        try:
            audit_exp(exp)
        except Exception as e:
            print(f"FAILED for {exp}: {e}")


if __name__ == "__main__":
    main()