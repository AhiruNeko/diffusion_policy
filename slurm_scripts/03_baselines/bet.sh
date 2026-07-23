#!/bin/sh
#SBATCH --job-name=dp_bet
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_bet_%j.out
# Phase 3: BET baseline | Table 4 | 预期: BlockPush p1=0.96, p2=0.71

source venv/bin/activate
RESULT_DIR=$(create_result_dir "03_baselines" "bet_blockpush" "bet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 training.rollout_every=100 training.checkpoint_every=100 checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score checkpoint.topk.mode=max checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "BET BlockPush" "0.96(p1)/0.71(p2)" "N/A"
