#!/bin/sh
#SBATCH --job-name=dp_square_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_square_image_%j.out
# Phase 4: Square image, DP-C | Table 2 | 预期: max=0.98, avg=0.92

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=square_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
