#!/bin/sh
#SBATCH --job-name=dp_t_pusht
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_t_pusht_%j.out
# Phase 5: Transformer DP-T, PushT low-dim | Table 1 | 预期: max=0.95, avg=0.79

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "05_transformer_ablation" "dp_transformer_pusht" "diffusion_transformer_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_transformer_lowdim_pusht_workspace.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 task.env_runner.n_test=15 training.rollout_every=100 training.checkpoint_every=100 checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score checkpoint.topk.mode=max checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "DP-T PushT" "0.95" "0.79"
