#!/bin/bash
#SBATCH --job-name=dp_multirun
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_multirun_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

python multi_run.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42,43,44 task.env_runner.n_envs=24 training.device=cuda:0
