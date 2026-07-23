#!/bin/sh
#SBATCH --job-name=dp_transport_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_transport_lowdim_%j.out
# Phase 2: Transport low-dim, DP-C | Table 1 | 预期: max=0.94, avg=0.82
# 注意: 需 32 CPU (n_envs=28), 必须服务器

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
