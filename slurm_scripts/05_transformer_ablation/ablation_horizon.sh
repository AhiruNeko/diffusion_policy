#!/bin/bash
#SBATCH --job-name=dp_ablation_horizon
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_ablation_horizon_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

for H in 4 8 16 32; do
    for A in 2 4 8 16; do
        [ $A -gt $H ] && continue
        TAG="${H}_${A}"
        python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
            horizon=${H} n_action_steps=${A} \
            training.seed=${SLURM_ARRAY_TASK_ID:-42} \
            task.env_runner.n_envs=1 training.device=cuda:0 \
            training.rollout_every=100 training.checkpoint_every=100 \
            checkpoint.topk.k=1 checkpoint.topk.monitor_key=test_mean_score \
            checkpoint.topk.mode=max checkpoint.save_last_ckpt=False \
            task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0
    done
done
