#!/bin/sh
#SBATCH --job-name=dp_pusht_image
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_pusht_image_%j.out

source venv/bin/activate
cd ~/projects/diffusion_policy
mkdir -p logs

python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0
