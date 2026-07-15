#!/bin/sh
#SBATCH --job-name=dp_pushT_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_pushT_lowdim_%j.out
# ====== Phase 1 — 核心 Low-dim ======
# 任务: PushT low-dim, DP-C  |  Table 1  |  预期: max=0.95, avg=0.91
# GPU建议: a100(~2h) / h20(~2h) / 4080(~4h) 均可
# 局部可跑 ✅ (~2GB VRAM)
# =====================================

GPU_TYPE="${GPU_TYPE:-a100}"

source slurm_scripts/00_env/config.sh
setup_env

RESULT_DIR=$(create_result_dir "01_lowdim_core" "pusht_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs

python train.py \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"

save_summary "$RESULT_DIR" "PushT low-dim" "0.95" "0.91"
