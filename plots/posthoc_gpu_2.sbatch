#!/bin/bash
#SBATCH --job-name=plot_angles
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load gcc python py-virtualenv
source ~/venvs/hgm/bin/activate

python /home/tabanell/hierarchical-model/plots/plot_principal_angles_ovA.py