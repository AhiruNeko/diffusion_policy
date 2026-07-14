#!/bin/bash
# ============================================================================
# Diffusion Policy 复现 — 通用环境配置脚本
# 支持 conda 和 venv 两种方式，自动检测
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/results}"

# ========== 环境激活（自动检测）==========
setup_env() {
    # 优先 conda
    if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q robodiff; then
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate robodiff
    # 其次 venv
    elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    # 初次使用：自动创建
    elif [ -f "$PROJECT_DIR/setup_venv.sh" ]; then
        echo "未找到环境，正在自动创建..."
        bash "$PROJECT_DIR/setup_venv.sh"
        source "$PROJECT_DIR/venv/bin/activate"
    else
        echo "错误：找不到 robodiff 环境"
        echo "请运行: bash $PROJECT_DIR/setup_venv.sh"
        exit 1
    fi
    echo "环境就绪: $(python --version), PyTorch $(python -c 'import torch; print(torch.__version__)')"
}

# ========== 结果目录 ==========
create_result_dir() {
    local phase=$1 task=$2 method=$3 seed=${4:-42}
    local dir="$RESULTS_BASE/$phase/${task}/${method}/seed_${seed}"
    mkdir -p "$dir"
    echo "$dir"
}

# ========== 保存结果 ==========
save_summary() {
    local result_dir=$1 task_name=$2 expected_max=$3 expected_avg=$4
    local ckpt_dir="$result_dir/checkpoints" summary_file="$result_dir/summary.txt"
    {
        echo "=========================================="
        echo "Diffusion Policy 复现结果"
        echo "任务: $task_name | 预期: max=$expected_max, avg=$expected_avg"
        echo "日期: $(date)"
        echo "=========================================="
        if [ -d "$ckpt_dir" ]; then
            ls "$ckpt_dir"/epoch=*-test_mean_score=*.ckpt 2>/dev/null | while read f; do
                echo "  $(basename $f)"
            done
            local scores=$(ls "$ckpt_dir"/epoch=*-test_mean_score=*.ckpt 2>/dev/null | \
                sed 's/.*test_mean_score=\([0-9.]*\).ckpt/\1/' | sort -n)
            if [ -n "$scores" ]; then
                local max_s=$(echo "$scores" | tail -1)
                local avg10=$(echo "$scores" | tail -10 | awk '{sum+=$1; n++} END {if(n>0) print sum/n}')
                echo "Max score: $max_s (预期: $expected_max)"
                echo "Avg last 10: $avg10 (预期: $expected_avg)"
            fi
        fi
    } > "$summary_file"
    cat "$summary_file"
}
