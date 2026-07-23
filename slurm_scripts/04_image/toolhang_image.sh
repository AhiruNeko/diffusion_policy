#!/bin/sh
#SBATCH --job-name=dp_toolhang_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_toolhang_image_%j.out
# Phase 4: Tool Hang image, DP-C | Table 2 | 预期: max=0.95, avg=0.73

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=tool_hang_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
