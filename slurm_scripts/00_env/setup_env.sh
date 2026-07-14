#!/bin/sh
#SBATCH --job-name=dp_setup_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dp_setup_env_%j.out
# ============================================
# 环境验证脚本
# 在服务器上联网安装 miniconda
# 只需跑一次
# ============================================

echo "===== 检查 conda ====="
if ! command -v conda &> /dev/null; then
    echo "安装 miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

echo "===== 检查 robodiff 环境 ====="
if ! conda env list | grep -q robodiff; then
    if [ -d "$HOME/robodiff_env" ]; then
        echo "从上传的目录克隆环境..."
        conda create --clone $HOME/robodiff_env -n robodiff -y
    fi
fi

conda activate robodiff
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo "===== 就绪 ====="
