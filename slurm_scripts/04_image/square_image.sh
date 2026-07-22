#!/bin/sh
#SBATCH --job-name=dp_square_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_square_image_%j.out
# Phase 4: Square image, DP-C | Table 2 | 预期: max=0.98, avg=0.92

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "04_image" "square_image" "diffusion_unet_hybrid" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=square_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 checkpoint.topk.k=1 checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Square image" "0.98" "0.92"
