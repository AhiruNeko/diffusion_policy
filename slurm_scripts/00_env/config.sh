#!/bin/bash
# ============================================================================
# Diffusion Policy 复现 — 通用环境配置脚本
# 服务器上无 conda，使用 Python venv 虚拟环境
# ============================================================================

# 项目根目录（自动检测）
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# venv 路径
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

# 结果根目录
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/results}"

# ========== 环境激活 ==========
setup_env() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "错误：找不到 venv 环境 $VENV_DIR"
        echo "请先运行: sbatch slurm_scripts/00_env/setup_env.sh"
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
    echo "环境就绪：$(python --version), PyTorch $(python -c 'import torch; print(torch.__version__)')"
}

# ========== 结果目录管理 ==========
create_result_dir() {
    local phase=$1
    local task=$2
    local method=$3
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
                [ -f "$ckpt" ] && echo "  $(basename $ckpt)"
            done
            echo ""
            local scores=$(ls "$ckpt_dir"/epoch=*-test_mean_score=*.ckpt 2>/dev/null | \
                sed 's/.*test_mean_score=\([0-9.]*\).ckpt/\1/' | sort -n)
            if [ -n "$scores" ]; then
                local max_score=$(echo "$scores" | tail -1)
                local last10=$(echo "$scores" | tail -10 | awk '{sum+=$1; n++} END {if(n>0) print sum/n}')
                echo "实际指标:"
                echo "  Max score:     $max_score  (预期: $expected_max)"
                echo "  Avg last 10:   $last10    (预期: $expected_avg)"
            fi
        else
            echo "  (训练未完成或无检查点)"
        fi
    } > "$summary_file"
    cat "$summary_file"
}
