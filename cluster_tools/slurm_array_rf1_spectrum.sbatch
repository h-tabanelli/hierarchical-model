#!/bin/bash
#SBATCH --job-name=rf1spec
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load gcc python py-virtualenv
source ~/venvs/hgm/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TASKFILE="${1}"
OUTDIR="${2}"
SPEC_RANK="${3:-256}"

python3 cluster_tools/run_rf1_spectrum.py \
  --taskfile "$TASKFILE" \
  --task_id ${SLURM_ARRAY_TASK_ID} \
  --outdir "$OUTDIR" \
  --spec_rank "$SPEC_RANK"