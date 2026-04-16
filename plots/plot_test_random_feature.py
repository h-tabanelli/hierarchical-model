import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl_glob(path_glob: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(path_glob, recursive=True):
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise ValueError(f"No JSON rows found for glob: {path_glob}")
    return pd.DataFrame(rows)


def plot_rf_two_panels(path_glob: str,
                       model: str = "true",
                       head_mode: str = "spectral_B",
                       layer1_mode: str = "rf_spectral"):
    df = load_jsonl_glob(path_glob)

    # filtres
    df = df[
        (df["model"] == model) &
        (df["head_mode"] == head_mode) &
        (df["layer1_mode"] == layer1_mode)
    ].copy()

    # déduplication
    dedup_cols = ["d", "alpha", "seed", "model", "head_mode", "layer1_mode"]
    for col in ["rf_width", "rf_activation"]:
        if col in df.columns:
            dedup_cols.append(col)

    df = df.sort_values(["d", "alpha", "seed"])
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    # nettoyage
    df["d"] = df["d"].astype(int)
    df["alpha"] = df["alpha"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    y_cols = ["nmse", "ovH"]
    pretty = {"nmse": "NMSE", "ovH": "Overlap on h"}

    ds = sorted(df["d"].unique())

    for ax, y in zip(axes, y_cols):
        for d in ds:
            sub = df[df["d"] == d].copy()
            g = sub.groupby("alpha")[y].agg(["mean", "std", "count"]).reset_index()
            sem = g["std"] / np.sqrt(np.maximum(g["count"], 1))

            ax.plot(g["alpha"], g["mean"], marker="o", label=f"d={d}")
            ax.fill_between(
                g["alpha"],
                g["mean"] - sem,
                g["mean"] + sem,
                alpha=0.15,
            )

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(pretty[y])
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    return fig, axes, df


EXP_GLOB = "results/rf_d*_final_calib/**/*.jsonl"

fig, axes, df = plot_rf_two_panels(EXP_GLOB)

Path("figures/rf").mkdir(parents=True, exist_ok=True)
plt.savefig("figures/rf/rf_all_d_two_panels.png", dpi=200, bbox_inches="tight")
plt.show()

# def plot_rf_run(
#     path_glob,
#     model="true",
#     head_mode="spectral_B",
#     layer1_mode="rf_spectral",
#     x_col="alpha",
#     y_cols=("nmse", "ovH", "corr_s"),
# ):
#     """
#     path_glob: ex
#       "results/**/metrics.jsonl"
#       or "results/rf_alpha3_check/**/*.jsonl"
#     """

#     files = list(Path(".").glob(path_glob))
#     rows = []

#     for f in files:
#         try:
#             with open(f, "r") as fh:
#                 for line in fh:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         rows.append(json.loads(line))
#                     except json.JSONDecodeError:
#                         pass
#         except Exception:
#             pass

#     if not rows:
#         raise ValueError(f"No JSON rows found for glob: {path_glob}")

#     df = pd.DataFrame(rows)

#     df = df.sort_values(["d", "alpha", "seed"])
#     df = df.drop_duplicates(
#         subset=["d", "alpha", "seed", "model", "head_mode", "layer1_mode"],
#         keep="last",
#     )

#     # optional filters
#     if "model" in df.columns:
#         df = df[df["model"] == model]
#     if "head_mode" in df.columns:
#         df = df[df["head_mode"] == head_mode]
#     if "layer1_mode" in df.columns:
#         df = df[df["layer1_mode"] == layer1_mode]

#     if df.empty:
#         raise ValueError("No rows left after filtering.")

#     df = df.sort_values([x_col, "seed"] if "seed" in df.columns else [x_col])

#     fig, axes = plt.subplots(1, len(y_cols), figsize=(5 * len(y_cols), 4))
#     if len(y_cols) == 1:
#         axes = [axes]

#     for ax, y in zip(axes, y_cols):
#         g = df.groupby(x_col)[y].agg(["mean", "std", "count"]).reset_index()
#         sem = g["std"] / np.sqrt(np.maximum(g["count"], 1))
#         ax.plot(g[x_col], g["mean"], marker="o")
#         ax.fill_between(g[x_col], g["mean"] - sem, g["mean"] + sem, alpha=0.2)
#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y)
#         # ax.grid(True, alpha=0.3)
#         pretty_names = {
#             "nmse": "NMSE",
#             "ovH": "Overlap on h",
#             "eig_corr_B": "Spectral corr. on B",
#             "nmse_scaled": "Scaled NMSE",
#         }
#         ax.set_title(pretty_names.get(y, y))
#         ax.set_ylim(-0.05, 1.05)

#     plt.tight_layout()
#     return df

# from pathlib import Path
# import matplotlib.pyplot as plt
# from plot_test_random_feature import plot_rf_run  # si la fonction est dans le meme fichier, ignore cette ligne

# EXP_GLOB = "results/rf_d80_alpha34_whiten_nocalib/**/*.jsonl"

# df = plot_rf_run(
#     EXP_GLOB,
#     model="true",
#     head_mode="spectral_B",
#     layer1_mode="rf_spectral",
#     y_cols=("nmse", "ovH", "eig_corr_B"),
# )

# plt.suptitle("RF spectral — d=80, 3 seeds", y=1.02)
# Path("figures").mkdir(exist_ok=True)
# plt.savefig("figures/rf/rf_d80_alpha34_whiten_nocalib.png", dpi=200, bbox_inches="tight")
# plt.show()