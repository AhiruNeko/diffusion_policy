#!/bin/sh
#SBATCH --job-name=dp_toolhang_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_toolhang_lowdim_%j.out
# Phase 2: Tool Hang low-dim, DP-C | Table 1 | 预期: max=0.50, avg=0.30 (较难)

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
