#!/bin/bash
# ====== Phase 2 — 扩展 Low-dim 批量提交 ======
# 这些任务可以并行运行
cd "$(dirname "$0")"
for f in *.slurm; do sbatch "$f"; done
