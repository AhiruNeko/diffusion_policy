#!/bin/bash
#SBATCH --job-name=dp_pushT_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_pushT_lowdim_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

export MUJOCO_GL=egl
echo "MUJOCO_GL=$MUJOCO_GL"

python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 task.env_runner.n_envs=1 training.device=cuda:0
