#!/bin/sh
#SBATCH --job-name=dp_lift_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_lift_image_%j.out
# Phase 4: Lift image, DP-C | Table 2 | 预期: max=1.00, avg=1.00

source venv/bin/activate
RESULT_DIR=$(create_result_dir "04_image" "lift_image" "diffusion_unet_hybrid" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
    task=lift_image_abs training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 training.rollout_every=100 training.checkpoint_every=100 checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score checkpoint.topk.mode=max checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Lift image" "1.00" "1.00"
