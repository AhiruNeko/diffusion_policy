#!/bin/sh
#SBATCH --job-name=dp_blockpush_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_blockpush_lowdim_%j.out
# Phase 2: BlockPush, DP-C | Table 4 | 预期: p1=0.36, p2=0.11

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "02_lowdim_extra" "blockpush_lowdim" "diffusion_unet_lowdim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_seed training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    training.device=cuda:0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "BlockPush low-dim" "0.36(p1)/0.11(p2)" "N/A"
