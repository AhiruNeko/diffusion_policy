#!/bin/bash
# ============================================================================
# Diffusion Policy 复现 — 通用环境配置脚本
# 作用：所有 Slurm 脚本在运行前 source 此文件以配置环境
# 用法：在 Slurm 脚本顶部 source config.sh
# ============================================================================

# ========== 用户需修改的配置 ==========
# Conda 安装位置（解压 robodiff_env.tar.gz 的目标路径）
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="robodiff"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/diffusion_policy}"

# 结果根目录
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/results}"

# 默认 GPU 类型（提交时可通过 --export=GPU_TYPE=a100 覆盖）
GPU_TYPE="${GPU_TYPE:-a100}"
PARTITION="${PARTITION:-gpu}"

# ========== 环境激活 ==========
setup_env() {
    # 如果 conda 命令不可用，手动初始化
    if ! command -v conda &> /dev/null; then
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "错误：找不到 conda，请设置 CONDA_BASE=$CONDA_BASE"
            exit 1
        fi
    fi

    # 如果环境不存在，从 tarball 解压
    if ! conda env list | grep -q "^$ENV_NAME "; then
        if [ -f "$PROJECT_DIR/robodiff_env.tar.gz" ]; then
            echo "解压预打包环境..."
            mkdir -p "$CONDA_BASE/envs/$ENV_NAME"
            tar -xzf "$PROJECT_DIR/robodiff_env.tar.gz" -C "$CONDA_BASE/envs/$ENV_NAME"
        else
            echo "错误：找不到 $PROJECT_DIR/robodiff_env.tar.gz"
            echo "请先将本地打包的环境文件传输到服务器"
            exit 1
        fi
    fi

    conda activate "$ENV_NAME"
    
    # 安装项目可编辑包
    cd "$PROJECT_DIR"
    pip install -e . -q 2>/dev/null
    
    echo "环境就绪：$(python --version), PyTorch $(python -c 'import torch; print(torch.__version__)')"
}

# ========== 结果目录管理 ==========
create_result_dir() {
    local phase=$1      # e.g., lowdim_core
    local task=$2       # e.g., pusht_lowdim
    local method=$3     # e.g., diffusion_unet_lowdim
    local seed=${4:-42}
    
    local dir="$RESULTS_BASE/$phase/${task}/${method}/seed_${seed}"
    mkdir -p "$dir"
    echo "$dir"
}

# ========== 保存结果摘要 ==========
save_summary() {
    local result_dir=$1
    local task_name=$2
    local expected_max=$3
    local expected_avg=$4
    
    # 从 checkpoint 文件名提取 test_mean_score
    local ckpt_dir="$result_dir/checkpoints"
    local summary_file="$result_dir/summary.txt"
    
    {
        echo "=========================================="
        echo "Diffusion Policy 复现结果"
        echo "任务: $task_name"
        echo "预期: max=$expected_max, avg=$expected_avg"
        echo "日期: $(date)"
        echo "=========================================="
        echo ""
        echo "检查点指标:"
        if [ -d "$ckpt_dir" ]; then
            for ckpt in "$ckpt_dir"/epoch=*-test_mean_score=*.ckpt; do
                if [ -f "$ckpt" ]; then
                    echo "  $(basename $ckpt)"
                fi
            done
            echo ""
            # 提取所有分数
            local scores=$(ls "$ckpt_dir"/epoch=*-test_mean_score=*.ckpt 2>/dev/null | \
                sed 's/.*test_mean_score=\([0-9.]*\).ckpt/\1/' | sort -n)
            if [ -n "$scores" ]; then
                local max_score=$(echo "$scores" | tail -1)
                local avg_score=$(echo "$scores" | awk '{sum+=$1; n++} END {if(n>0) print sum/n}')
                local last10=$(echo "$scores" | tail -10 | awk '{sum+=$1; n++} END {if(n>0) print sum/n}')
                echo "实际指标:"
                echo "  Max score:     $max_score  (预期: $expected_max)"
                echo "  Avg last 10:   $last10    (预期: $expected_avg)"
                echo "  Overall avg:   $avg_score"
            fi
        else
            echo "  (训练未完成或无检查点)"
        fi
    } > "$summary_file"
    
    cat "$summary_file"
}

# ========== 日志记录 ==========
log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
