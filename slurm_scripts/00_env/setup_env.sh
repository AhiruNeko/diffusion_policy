#!/bin/sh
#SBATCH --job-name=dp_setup_env
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dp_setup_env_%j.out
# ============================================
# 环境初始化脚本
# 作用：在 GPU 节点上解压 conda 环境
# 只需在第一次使用前跑一次
# 用法：sbatch slurm_scripts/00_env/setup_env.sh
# ============================================

echo "===== 开始配置环境 ====="
echo "主机名: $(hostname)"
echo "用户: $USER"
echo "工作目录: $(pwd)"

# 设置 Conda 路径
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="robodiff"
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"
TARBALL="robodiff_env.tar.gz"

# 1. 检查环境包是否存在
if [ ! -f "$TARBALL" ]; then
    echo "错误：找不到 $TARBALL"
    echo "请确认压缩包在项目根目录"
    exit 1
fi

# 2. 解压环境
echo "解压环境到: $ENV_PATH"
mkdir -p "$ENV_PATH"
tar -xzf "$TARBALL" -C "$ENV_PATH"
echo "环境解压完成"

# 3. 验证
echo "===== 验证环境 ====="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -c "
import torch
import gym
import mujoco_py
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('MuJoCo:', mujoco_py.__version__)
"

echo "===== 环境配置完成 ====="
echo "现在可以提交训练脚本了"
