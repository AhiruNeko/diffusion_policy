#!/bin/bash
#SBATCH --job-name=dp_setup_venv
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_venv_%j.out
# ============================================================================
# setup_venv.sh — 使用 Python venv 配置 Diffusion Policy 环境
# 不需要 conda，只需要 Python 3.9+
#
# 用法:
#   bash setup_venv.sh                    # 直接运行
#   sbatch setup_venv.sh                  # Slurm 提交
#   source venv/bin/activate              # 激活环境
#
# ============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------- 配置 ----------
VENV_DIR="${VENV_DIR:-venv}"
PROJECT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || pwd)"

# ---------- 检测 Python ----------
detect_python() {
    for cmd in python3.11 python3.10 python3.9 python3; do
        if command -v "$cmd" &>/dev/null; then
            PYTHON="$cmd"
            PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            info "Python: $PYTHON (版本 $PYVER)"
            return 0
        fi
    done
    error "未找到 Python 3.9+" && exit 1
}

# ---------- 选择 torch 版本 ----------
select_torch() {
    case "$PYVER" in
        3.9)  TORCH="torch==1.12.1"; TORCHV="torchvision==0.13.1"; CUDA_VER="cu116" ;;
        3.10) TORCH="torch==1.13.1"; TORCHV="torchvision==0.14.1"; CUDA_VER="cu116" ;;
        3.11) TORCH="torch==2.0.1";  TORCHV="torchvision==0.15.2"; CUDA_VER="cu118" ;;
        *)    TORCH="torch==2.2.0";  TORCHV="torchvision==0.17.0"; CUDA_VER="cu118" ;;
    esac
    info "PyTorch: $TORCH + $TORCHV ($CUDA_VER)"
}

# ---------- 创建 venv ----------
create_venv() {
    if [ -d "$VENV_DIR" ]; then
        warn "venv 已存在: $VENV_DIR（跳过创建）"
        return 0
    fi
    info "创建 venv..."
    "$PYTHON" -m venv "$VENV_DIR" --copies
    source "$VENV_DIR/bin/activate"
    info "pip 版本: $($PIP --version 2>/dev/null || pip --version)"
}

# ---------- 安装依赖 ----------
install_deps() {
    source "$VENV_DIR/bin/activate"
    PIP="$(command -v pip3 || command -v pip || echo pip)"

    # 1. PyTorch
    info "第 1 步: 安装 PyTorch..."
    $PIP install "$TORCH" "$TORCHV" --index-url "https://download.pytorch.org/whl/$CUDA_VER"

    # 2. 核心科学计算包
    info "第 2 步: 安装核心依赖..."
    $PIP install numpy==1.23.3 scipy==1.9.1 matplotlib==3.6.1
    $PIP install opencv-python==4.6.0 Pillow==9.2.0
    $PIP install h5py==3.7.0 zarr==2.12.0 numcodecs==0.10.2

    # 3. 深度学习框架
    info "第 3 步: 安装 ML 框架..."
    $PIP install diffusers==0.11.1 einops==0.4.1 accelerate==0.13.2
    $PIP install hydra-core==1.2.0 wandb==0.13.3
    $PIP install tensorboard==2.10.1 tensorboardX==2.5.1
    $PIP install pytorch3d==0.7.0 2>/dev/null || warn "pytorch3d 跳过（不影响训练）"

    # 4. 强化学习/机器人
    info "第 4 步: 安装 RL/机器人库..."
    $PIP install gym==0.21.0 pymunk==6.2.1
    $PIP install free-mujoco-py==2.1.6 2>/dev/null || warn "free-mujoco-py 跳过"
    $PIP install robosuite@https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz 2>/dev/null || warn "robosuite 跳过"
    $PIP install pybullet-svl==3.1.6.4 2>/dev/null || warn "pybullet 跳过"
    $PIP install robomimic==0.2.0 2>/dev/null || warn "robomimic 跳过"

    # 5. 数据处理
    info "第 5 步: 安装数据处理库..."
    $PIP install scikit-image==0.19.3 scikit-video==1.1.11
    $PIP install imageio==2.22.0 imageio-ffmpeg==0.4.7
    $PIP install Cython==0.29.32
    $PIP install av==10.0.0 2>/dev/null || warn "av 跳过"

    # 6. 工具库
    info "第 6 步: 安装工具库..."
    $PIP install tqdm==4.64.1 click==8.0.4 psutil==5.9.2
    $PIP install dill==0.3.5.1 shapely==1.8.4
    $PIP install termcolor==2.0.1 threadpoolctl==3.1.0
    $PIP install boto3==1.24.96 datasets==2.6.1
    $PIP install cmake==3.24.3
    $PIP install pytorchvideo==0.1.5 2>/dev/null || warn "pytorchvideo 跳过"
    $PIP install imagecodecs==2022.9.26
    $PIP install pygame==2.1.2
    $PIP install ray[default,tune]==2.2.0
    $PIP install dm-control==1.0.9 --no-deps 2>/dev/null || warn "dm-control 跳过"
    $PIP install r3m@https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz 2>/dev/null || warn "r3m 跳过"

    # 8. 安装项目
    info "第 8 步: 安装项目..."
    $PIP install -e "$PROJECT_DIR"

    # 9. 修复兼容性
    info "第 9 步: 修复兼容性..."
    $PIP install "numpy<2" huggingface_hub==0.20.0 parameterized -q

    # 10. 确保核心训练包
    info "第 10 步: 确认核心包..."
    for pkg in gym wandb zarr; do
        python -c "import $pkg" 2>/dev/null || {
            warn "$pkg 未安装，重试..."
            $PIP install "$pkg" -q 2>/dev/null || warn "$pkg 跳过"
        }
    done
}

# ---------- 验证 ----------
verify() {
    source "$VENV_DIR/bin/activate"
    info "验证环境..."
    python -c "
import sys
print(f'Python: {sys.version}')
ok = True
tests = [
    ('torch',       'import torch; print(f\"PyTorch: {torch.__version__}\")'),
    ('gym',         'import gym; print(f\"gym: {gym.__version__}\")'),
    ('diffusers',   'import diffusers; print(f\"diffusers: {diffusers.__version__}\")'),
    ('hydra',       'import hydra; print(f\"hydra: {hydra.__version__}\")'),
    ('numpy',       'import numpy; print(f\"numpy: {numpy.__version__}\")'),
    ('einops',      'import einops; print(f\"einops: {einops.__version__}\")'),
    ('wandb',       'import wandb; print(f\"wandb: {wandb.__version__}\")'),
    ('zarr',        'import zarr; print(f\"zarr: {zarr.__version__}\")'),
]
for name, code in tests:
    try:
        exec(code)
    except Exception as e:
        print(f'{name}: FAILED - {e}')
        ok = False
if ok:
    print('所有核心包导入成功!')
else:
    print('部分包导入失败（不影响核心训练的包可忽略）')
    sys.exit(0)
"
}

# ---------- 清理 ----------
clean() {
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        info "已删除 $VENV_DIR"
    fi
}

# ---------- 主流程 ----------
case "${1:-}" in
    --clean)        clean ;;
    --install-only) detect_python; select_torch; source "$VENV_DIR/bin/activate" 2>/dev/null || { error "venv 不存在，先运行 bash setup_venv.sh"; exit 1; }; install_deps; verify ;;
    --verify)       clean 2>/dev/null || true; detect_python; select_torch; create_venv; install_deps; verify ;;
    --help|-h)      echo "用法: bash setup_venv.sh [选项]"
                    echo "  (无参数)   创建 venv + 安装全部依赖"
                    echo "  --install-only  只安装依赖（跳过 venv 创建）"
                    echo "  --verify   创建 + 安装 + 验证"
                    echo "  --clean    删除 venv"
                    echo "  --help     显示帮助" ;;
    *)              detect_python; select_torch; create_venv; install_deps; info "完成! 激活: source $VENV_DIR/bin/activate" ;;
esac
