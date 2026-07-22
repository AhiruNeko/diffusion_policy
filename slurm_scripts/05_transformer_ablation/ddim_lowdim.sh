#!/bin/sh
#SBATCH --job-name=dp_ddim_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_ddim_lowdim_%j.out
# Phase 5: DDIM 快速采样消融

source slurm_scripts/00_env/config.sh
setup_env
RESULT_DIR=$(create_result_dir "05_transformer_ablation" "ddim_lowdim_pusht" "diffusion_unet_ddim" "${SLURM_ARRAY_TASK_ID:-42}")
mkdir -p logs
HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_ddim_lowdim_workspace.yaml \
    training.seed=${SLURM_ARRAY_TASK_ID:-42} training.device=cuda:0 checkpoint.topk.k=1 checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 \
    hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
save_summary "$RESULT_DIR" "DDIM lowdim PushT" "见论文Fig5" "见论文Fig5"
