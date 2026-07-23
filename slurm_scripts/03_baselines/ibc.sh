#!/bin/bash
#SBATCH --job-name=dp_ibc
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_ibc_%j.out

TASK="${TASK:-pusht_lowdim}"

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

python train.py --config-name=train_ibc_dfo_lowdim_workspace.yaml \
    task=${TASK} training.seed=${SLURM_ARRAY_TASK_ID:-42} \
    task.env_runner.n_envs=1 training.device=cuda:0
