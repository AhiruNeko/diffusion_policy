#!/bin/bash
# ============================================================================
# Diffusion Policy 复现 — 通用环境配置脚本
# 固定使用项目目录下的 venv
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/results}"

setup_env() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "错误: 找不到 $VENV_DIR/bin/activate"
        echo "请先运行: bash setup_venv.sh"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
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
