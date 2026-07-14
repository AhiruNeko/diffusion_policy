#!/bin/sh
#SBATCH --job-name=dp_setup_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dp_setup_env_%j.out
# ============================================
# 环境配置脚本（sbatch 提交版）
# 使用 Python venv，不需要 conda
# 只需提交一次，等它跑完即可
# 用法：sbatch slurm_scripts/00_env/setup_env.sh
# ============================================

echo "===== 开始配置 Diffusion Policy 环境 ====="
echo "主机名: $(hostname)"
echo "Python: $(python3 --version)"

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/venv"

# ---------- 1. 检测 Python ----------
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python 3.9+"
    exit 1
fi
PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "使用: $PYTHON (版本 $PYVER)"

# ---------- 2. 选择 torch ----------
case "$PYVER" in
    3.9)  TORCH="torch==1.12.1";   TORCHV="torchvision==0.13.1"; CUDA="cu116" ;;
    3.10) TORCH="torch==1.13.1";   TORCHV="torchvision==0.14.1"; CUDA="cu116" ;;
    3.11) TORCH="torch==2.0.1";    TORCHV="torchvision==0.15.2"; CUDA="cu118" ;;
    *)    TORCH="torch==2.2.0";    TORCHV="torchvision==0.17.0"; CUDA="cu118" ;;
esac
echo "PyTorch: $TORCH + $TORCHV ($CUDA)"

# ---------- 3. 创建 venv ----------
echo "创建 venv..."
$PYTHON -m venv "$VENV_DIR" --copies
source "$VENV_DIR/bin/activate"

# ---------- 4. 安装依赖 ----------
PIP="$(command -v pip3 || command -v pip)"

echo "[1/8] PyTorch..."
$PIP install "$TORCH" "$TORCHV" --index-url "https://download.pytorch.org/whl/$CUDA"

echo "[2/8] 核心科学计算..."
$PIP install numpy==1.23.3 scipy==1.9.1 matplotlib==3.6.1 opencv-python==4.6.0
$PIP install Pillow==9.2.0 h5py==3.7.0 zarr==2.12.0 numcodecs==0.10.2

echo "[3/8] 深度学习框架..."
$PIP install diffusers==0.11.1 einops==0.4.1 accelerate==0.13.2
$PIP install hydra-core==1.2.0 wandb==0.13.3 tensorboard==2.10.1 tensorboardX==2.5.1

echo "[4/8] RL/机器人库..."
$PIP install gym==0.21.0 pymunk==6.2.1
$PIP install free-mujoco-py==2.1.6 2>/dev/null || echo "  free-mujoco-py 跳过"
$PIP install pybullet-svl==3.1.6.4 2>/dev/null || echo "  pybullet 跳过"
$PIP install robomimic==0.2.0 2>/dev/null || echo "  robomimic 跳过"

echo "[5/8] 数据处理..."
$PIP install scikit-image==0.19.3 scikit-video==1.1.11
$PIP install imageio==2.22.0 imageio-ffmpeg==0.4.7 Cython==0.29.32
$PIP install av==10.0.0 2>/dev/null || echo "  av 跳过"

echo "[6/8] 工具库..."
$PIP install tqdm==4.64.1 click==8.0.4 psutil==5.9.2 dill==0.3.5.1
$PIP install shapely==1.8.4 termcolor==2.0.1 threadpoolctl==3.1.0
$PIP install boto3==1.24.96 datasets==2.6.1 cmake==3.24.3
$PIP install imagecodecs==2022.9.26 pygame==2.1.2 ray[default,tune]==2.2.0
$PIP install pytorchvideo==0.1.5 2>/dev/null || echo "  pytorchvideo 跳过"

echo "[7/8] 项目本身..."
$PIP install -e "$PROJECT_DIR"

echo "[8/8] 修复兼容性..."
$PIP install "numpy<2" huggingface_hub==0.20.0 parameterized -q

# 补装核心包（如果上面某步失败了）
for pkg in gym wandb zarr; do
    python -c "import $pkg" 2>/dev/null || { $PIP install "$pkg" -q; }
done

# ---------- 5. 验证 ----------
echo ""
echo "===== 验证 ====="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import gym; print('gym:', gym.__version__)
import diffusers; print('diffusers:', diffusers.__version__)
import hydra; print('hydra:', hydra.__version__)
import wandb; print('wandb:', wandb.__version__)
import zarr; print('zarr:', zarr.__version__)
print('全部核心包 OK!')
"

echo ""
echo "===== 环境配置完成 ====="
echo "激活: source $VENV_DIR/bin/activate"
echo "训练: sbatch slurm_scripts/01_lowdim_core/pushT_lowdim.sh"
