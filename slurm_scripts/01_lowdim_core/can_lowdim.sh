#!/bin/sh
#SBATCH --job-name=dp_can_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dp_can_lowdim_%j.out
# Phase 1: Can low-dim, DP-C | Table 1 | 预期: max=1.00, avg=0.96

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "01_lowdim_core" "can_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Can low-dim" "1.00" "0.96"
