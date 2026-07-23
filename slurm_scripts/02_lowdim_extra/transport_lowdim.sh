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
RESULT_DIR=$(create_result_dir "02_lowdim_extra" "transport_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 training.rollout_every=100 training.checkpoint_every=100 checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score checkpoint.topk.mode=max checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Transport low-dim" "0.94" "0.82"
