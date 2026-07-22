#!/bin/sh
#SBATCH --job-name=dp_ablation_horizon
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_ablation_horizon_%j.out
# Phase 5: 动作视界消融 (horizon=4,8,16,32) | 对应论文 Fig 4

source slurm_scripts/00_env/config.sh
setup_env

for H in 4 8 16 32; do
    for A in 2 4 8 16; do
        [ $A -gt $H ] && continue
        TAG="horizon${H}_action${A}"
        RESULT_DIR=$(create_result_dir "05_transformer_ablation" "ablation_horizon" "${TAG}" "${SLURM_ARRAY_TASK_ID:-42}")
        HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
            horizon=${H} n_action_steps=${A} training.seed=${SLURM_ARRAY_TASK_ID:-42} \
            training.device=cuda:0 task.env_runner.n_test=15 training.rollout_every=100 training.checkpoint_every=100 checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score checkpoint.topk.mode=max checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0 hydra.run.dir="$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/train.log"
    done
done
