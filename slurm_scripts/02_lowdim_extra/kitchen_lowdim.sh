#!/bin/sh
#SBATCH --job-name=dp_kitchen_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_kitchen_lowdim_%j.out
# Phase 2: Kitchen, DP-C | Table 4 | 预期: 4st 1.00/1.00/1.00/0.99

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
