#!/bin/sh
#SBATCH --job-name=dp_setup_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dp_setup_env_%j.out
# ============================================
# 环境初始化脚本（纯 venv 版）
# 不需要 conda，用 Python 自带的 venv
# 用法：sbatch slurm_scripts/00_env/setup_env.sh
# ============================================

echo "===== 开始配置环境 ====="
echo "主机名: $(hostname)"
echo "Python: $(python3 --version)"

# 1. 创建虚拟环境
echo "创建 venv 虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 2. 安装 PyTorch（单独指定 CUDA 版本）
echo "安装 PyTorch..."
pip install torch==1.12.1 torchvision==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu116

# 3. 安装所有依赖
echo "安装其余依赖..."
pip install -r requirements_frozen.txt

# 4. 安装项目本身
pip install -e .

# 5. 验证
echo "===== 验证 ====="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
import gym; print('gym:', gym.__version__)
import mujoco_py; print('MuJoCo:', mujoco_py.__version__)
"

echo "===== 完成 ====="
echo "激活方式: source venv/bin/activate"
