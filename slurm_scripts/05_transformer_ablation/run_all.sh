#!/bin/bash
# ====== Phase 5 — Transformer / 消融实验批量提交 ======
cd "$(dirname "$0")"
for f in *.slurm; do sbatch "$f"; done
