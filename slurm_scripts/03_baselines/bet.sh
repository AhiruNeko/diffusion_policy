#!/bin/bash
#SBATCH --job-name=dp_bet
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_bet_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

python train.py --config-name=train_bet_lowdim_workspace.yaml \
    task=blockpush_lowdim_seed training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    task.env_runner.n_envs=1 training.device=cuda:0
