#!/bin/sh
#SBATCH --job-name=dp_kitchen_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dp_kitchen_lowdim_%j.out
# Phase 2: Kitchen, DP-C | Table 4 | 预期: 4st 1.00/1.00/1.00/0.99

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "02_lowdim_extra" "kitchen_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=kitchen_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Kitchen low-dim" "4st:1.00/1.00/1.00/0.99" "N/A"
