#!/bin/bash
#SBATCH --job-name=hung16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc python py-virtualenv
source ~/venvs/hgm/bin/activate

python3 plots/plot_hungarian_matched_directions.py