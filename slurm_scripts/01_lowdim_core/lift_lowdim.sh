#!/bin/bash
#SBATCH --job-name=dp_lift_lowdim
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_lift_lowdim_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "MUJOCO_GL=$MUJOCO_GL"

python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim training.seed=42 task.env_runner.n_envs=1 training.device=cuda:0 checkpoint.topk.k=1 checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0
