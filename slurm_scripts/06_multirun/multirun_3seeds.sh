#!/bin/sh
#SBATCH --job-name=dp_multirun
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx4080:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/dp_multirun_%j.out
# Phase 6: 多种子训练 (3 seeds: 42,43,44) | 需 3 张 GPU 并行

source venv/bin/activate

cd "$PROJECT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3

python ray_train_multirun.py \
    --config-dir=. \
    --config-name=image_pusht_diffusion_policy_cnn.yaml \
    --seeds=42,43,44 \
    --monitor_key=test/mean_score \
    -- multi_run.run_dir="$RESULTS_BASE/06_multirun/pusht_image_3seeds/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}"

python multirun_metrics.py --input_dir "$RESULTS_BASE/06_multirun/pusht_image_3seeds"
ray stop
