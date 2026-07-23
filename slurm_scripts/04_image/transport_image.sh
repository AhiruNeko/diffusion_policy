#!/bin/sh
#SBATCH --job-name=dp_transport_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_transport_image_%j.out
# Phase 4: Transport image, DP-C | Table 2 | 预期: max=1.00, avg=0.93
# 注意: 需 32 CPU + 96G 内存, 必须服务器

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=transport_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
