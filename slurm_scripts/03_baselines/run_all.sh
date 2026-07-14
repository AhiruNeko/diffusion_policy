#!/bin/bash
# ====== Phase 3 — 基线方法批量提交 ======
cd "$(dirname "$0")"
for f in *.slurm; do sbatch "$f"; done
