#!/bin/sh
#SBATCH --job-name=dp_pusht_image
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dp_pusht_image_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

MUJOCO_GL=egl python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 task.env_runner.n_envs=1 training.device=cuda:0 task.env_runner.n_test=15 checkpoint.topk.k=1 checkpoint.save_last_ckpt=False task.env_runner.n_test_vis=1 task.env_runner.n_train_vis=0
