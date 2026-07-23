#!/bin/sh
#SBATCH --job-name=dp_blockpush_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_blockpush_lowdim_%j.out
# Phase 2: BlockPush, DP-C | Table 4 | 预期: p1=0.36, p2=0.11

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_seed training.seed=${SLURM_ARRAY_TASK_ID:-42} \
