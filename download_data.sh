#!/bin/bash
# ============================================================================
# download_data.sh — 下载 Diffusion Policy 复现所需数据集
# 用法:
#   bash download_data.sh                         # 下载全部
#   bash download_data.sh --only-lowdim            # 只下 lowdim（必跑实验）
#   bash download_data.sh --only-image             # 只下 image（可选）
#   bash download_data.sh --check                  # 只检查，不下载
# ============================================================================

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
BASE_URL="https://diffusion-policy.cs.columbia.edu/data/training"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 数据清单
declare -A DATASETS
DATASETS[robomimic_lowdim]="robomimic_lowdim.zip:2.9GB:Square, Can, Lift, Transport, ToolHang (lowdim) ✅ 必跑"
DATASETS[robomimic_image]="robomimic_image.zip:1.5GB:Square, Can, Lift, Transport, ToolHang (image) 🔶 可选"
DATASETS[block_pushing]="block_pushing.zip:12MB:BlockPush (BET baseline) ✅ 建议跑"
DATASETS[kitchen]="kitchen.zip:1.8GB:Kitchen task 🔶 可选"

# ---------- 检查 ----------
check() {
    info "检查已有数据..."
    for key in "${!DATASETS[@]}"; do
        IFS=':' read -r fname size desc <<< "${DATASETS[$key]}"
        local ok=false
        case "$key" in
            robomimic_lowdim) [ -d "robomimic/datasets/square/ph" ] && [ -f "robomimic/datasets/square/ph/low_dim.hdf5" ] && ok=true ;;
            robomimic_image)  [ -d "robomimic/datasets/square/ph" ] && [ -f "robomimic/datasets/square/ph/image_abs.hdf5" ] && ok=true ;;
            block_pushing)    [ -f "block_pushing/multimodal_push_seed.zarr/.zgroup" ] && ok=true ;;
            kitchen)          [ -f "kitchen/all_actions.npy" ] && ok=true ;;
        esac
        if $ok; then
            echo -e "  ✅ $key ($size) — 已就绪"
        else
            echo -e "  ❌ $key ($size) — 缺失"
        fi
    done
}

# ---------- 下载单个 ----------
download_one() {
    local key=$1
    IFS=':' read -r fname size desc <<< "${DATASETS[$key]}"

    # 检查是否已有
    case "$key" in
        robomimic_lowdim) [ -f "robomimic/datasets/square/ph/low_dim.hdf5" ] && info "$key 已存在，跳过" && return 0 ;;
        robomimic_image)  [ -f "robomimic/datasets/square/ph/image_abs.hdf5" ] && info "$key 已存在，跳过" && return 0 ;;
        block_pushing)    [ -f "block_pushing/multimodal_push_seed.zarr/.zgroup" ] && info "$key 已存在，跳过" && return 0 ;;
        kitchen)          [ -f "kitchen/all_actions.npy" ] && info "$key 已存在，跳过" && return 0 ;;
    esac

    info "下载 $fname ($size) — $desc"
    wget -c "$BASE_URL/$fname" -O "$fname.tmp" 2>&1
    if [ $? -ne 0 ]; then
        error "下载失败: $fname"
        rm -f "$fname.tmp"
        return 1
    fi
    mv "$fname.tmp" "$fname"

    info "解压 $fname..."
    unzip -o "$fname" 2>&1 | tail -3
    rm -f "$fname"
    info "完成: $key"
}

# ---------- 全部下载 ----------
download_all() {
    for key in "${!DATASETS[@]}"; do
        download_one "$key"
    done
}

# ---------- 主流程 ----------
case "${1:-}" in
    --check)   check ;;
    --only-lowdim)
        download_one robomimic_lowdim
        download_one block_pushing
        download_one kitchen
        check
        ;;
    --only-image)
        download_one robomimic_image
        check
        ;;
    *)
        download_all
        check
        ;;
esac

echo ""
info "数据目录: $DATA_DIR"
echo "  pusht/               — 已有 (gitignored, 随项目一起传)"
echo "  其余数据下载完成即可使用"
