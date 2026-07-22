#!/bin/sh
#SBATCH --job-name=dp_ibc
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_ibc_%j.out
# Phase 3: IBC baseline | Table 1 | 预期: PushT=0.90/0.84, Square/Can/Lift=0.00
# 需要手动改 TASK 变量切换任务: pusht_lowdim / square_lowdim / can_lowdim / lift_lowdim

TASK="${TASK:-pusht_lowdim}"

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "03_baselines" "ibc_${TASK}" "ibc_dfo_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
HYDRA_FULL_ERROR=1 python train.py --config-name=train_ibc_dfo_lowdim_workspace.yaml \
    task=${TASK} training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 checkpoint.topk.k=1 checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "IBC ${TASK}" "见表1" "见表1"
