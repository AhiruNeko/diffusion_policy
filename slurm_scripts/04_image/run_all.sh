#!/bin/bash
# ====== Phase 4 — Image 任务批量提交 ======
# 注意: 这些任务需要 8GB+ VRAM
# 建议使用 A100(6h) / H20(6h) / 4080(12h)
cd "$(dirname "$0")"
for f in *.slurm; do sbatch "$f"; done
