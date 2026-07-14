#!/bin/sh
#SBATCH --job-name=dp_bcrnn
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dp_bcrnn_%j.out
# Phase 3: BC-RNN baseline | Table 1 | 预期: Square=0.95/0.73, Can=1.00/0.91, Lift=1.00/0.96
# 需要手动改 TASK 变量

TASK="${TASK:-square_lowdim}"

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "03_baselines" "bcrnn_${TASK}" "robomimic_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_robomimic_lowdim_workspace.yaml \
    task=${TASK} training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "BC-RNN ${TASK}" "见表1" "见表1"
