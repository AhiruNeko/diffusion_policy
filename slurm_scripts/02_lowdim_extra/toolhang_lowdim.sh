#!/bin/sh
#SBATCH --job-name=dp_toolhang_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_toolhang_lowdim_%j.out
# Phase 2: Tool Hang low-dim, DP-C | Table 1 | 预期: max=0.50, avg=0.30 (较难)

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "02_lowdim_extra" "toolhang_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "Tool Hang low-dim" "0.50" "0.30"
