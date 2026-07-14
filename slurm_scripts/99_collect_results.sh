#!/bin/bash
# ====== 结果汇总脚本 ======
# 用法: bash slurm_scripts/99_collect_results.sh
# 在服务器上运行，汇总所有实验结果

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
RESULTS_DIR="$PROJECT_DIR/results"
SUMMARY_FILE="$RESULTS_DIR/FINAL_SUMMARY.md"

cat > "$SUMMARY_FILE" << 'EOF'
# Diffusion Policy 复现结果汇总
| 任务 | 方法 | 我的 max | 我的 avg(last10) | 论文 max | 论文 avg | 匹配? |
|------|------|---------|-----------------|---------|---------|-------|
EOF

for phase_dir in "$RESULTS_DIR"/*/; do
    phase=$(basename "$phase_dir")
    for task_dir in "$phase_dir"*/; do
        task=$(basename "$task_dir")
        for method_dir in "$task_dir"*/; do
            method=$(basename "$method_dir")
            for seed_dir in "$method_dir"*/; do
                summary="$seed_dir/summary.txt"
                if [ -f "$summary" ]; then
                    my_max=$(grep "Max score:" "$summary" | awk '{print $NF}')
                    my_avg=$(grep "Avg last 10:" "$summary" | awk '{print $NF}')
                    expected=$(grep "预期:" "$summary" | head -1)
                    echo "| $task | $method | $my_max | $my_avg | $expected | |" >> "$SUMMARY_FILE"
                fi
            done
        done
    done
done

echo "汇总已保存至: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
