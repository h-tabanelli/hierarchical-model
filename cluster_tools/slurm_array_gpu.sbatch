#!/bin/bash
#SBATCH --job-name=hgm2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 cluster_tools/run_task_2layers.py \
  --taskfile tasks.jsonl \
  --task_id ${SLURM_ARRAY_TASK_ID} \
  --outdir results