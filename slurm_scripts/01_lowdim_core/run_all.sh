#!/bin/bash
# ============================================================================
# Phase 1 — 核心 Low-dim 训练脚本合集
# 对应论文 Table 1 (State Observation) — DP-C 部分
# 批量提交全部 4 个并行任务
# ============================================================================
# 用法: sbatch slurm_scripts/01_lowdim_core/run_all.sh
#       或单独提交每个任务
# ============================================================================

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
for script in "$CURRENT_DIR"/*.slurm; do
    [ "$(basename "$script")" = "run_all.sh" ] && continue
    echo "提交: $(basename "$script")"
    sbatch "$script"
done
echo "全部 Phase 1 任务已提交"
