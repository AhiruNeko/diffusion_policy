#!/bin/sh
#SBATCH --job-name=dp_can_image
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dp_can_image_%j.out
# Phase 4: Can image, DP-C | Table 2 | 预期: max=1.00, avg=0.97

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "04_image" "can_image" "diffusion_unet_hybrid" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=can_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Can image" "1.00" "0.97"
