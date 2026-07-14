#!/bin/sh
#SBATCH --job-name=dp_pusht_image
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dp_pusht_image_%j.out
# Phase 4: PushT image, DP-C | Table 2 | 预期: max=0.91, avg=0.84
# VRAM ~7GB, 建议 a100(6h) / h20(6h) / 4080(12h)

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "04_image" "pusht_image" "diffusion_unet_hybrid" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "PushT image" "0.91" "0.84"
