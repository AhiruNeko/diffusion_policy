#!/bin/sh
#SBATCH --job-name=dp_t_pusht
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dp_t_pusht_%j.out
# Phase 5: Transformer DP-T, PushT low-dim | Table 1 | 预期: max=0.95, avg=0.79

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "05_transformer_ablation" "dp_transformer_pusht" "diffusion_transformer_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_transformer_lowdim_pusht_workspace.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "DP-T PushT" "0.95" "0.79"
