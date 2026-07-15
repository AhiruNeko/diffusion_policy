#!/bin/sh
#SBATCH --job-name=dp_bet
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_bet_%j.out
# Phase 3: BET baseline | Table 4 | 预期: BlockPush p1=0.96, p2=0.71

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "03_baselines" "bet_blockpush" "bet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
HYDRA_FULL_ERROR=1 python train.py --config-name=train_bet_lowdim_workspace.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "BET BlockPush" "0.96(p1)/0.71(p2)" "N/A"
