# Diffusion Policy 论文复现指南

**论文**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2303.04137v5)

---

## 1. 前置条件

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
cd ~/projects/diffusion_policy
```

---

## 2. 论文核心表格解读与复现重点

### Table 1 — State Observation（低维观测，最重要）

| 任务 | BC-RNN | IBC | BET | **DP-C (CNN)** ⭐ | DP-T (Transformer) |
|------|--------|-----|-----|-----------------|-------------------|
| **PushT** | 0.84/0.76 | 0.90/0.84 | 0.00/0.00 | **0.95/0.91** | 0.95/0.79 |
| **Square** | 0.95/0.73 | 0.00/0.00 | 0.72/0.60 | **1.00/0.93** | 0.55/0.46 |
| **Can** | 0.96/0.93 | 0.00/0.00 | 0.80/0.73 | **1.00/0.96** | 0.95/0.93 |
| **Lift** | 1.00/1.00 | 0.49/0.28 | 0.98/0.96 | **1.00/0.98** | 1.00/0.95 |
| Transport | 0.65/0.52 | 0.00/0.00 | 0.67/0.52 | **0.94/0.82** | 0.92/0.86 |
| ToolHang | 0.00/0.00 | 0.00/0.00 | 0.19/0.08 | **0.50/0.30** | 0.45/0.35 |

**格式**: `max成功率 / 最后10个检查点平均成功率`
**重点解读**:
- DP-C（CNN）在 **全部 6 个任务上碾压所有基线**
- IBC 极不稳定：Square/Can/Transport/ToolHang 全部 0.00
- BET 在 PushT 上也是 0.00（无法建模连续动作）
- **你的复现重点**：🟢 PushT, Square, Can, Lift（DP-C 接近满分，验证算法有效）

### Table 2 — Visual Policy（图像观测）

| 任务 | BC-RNN | IBC | BET | **DP-C (CNN)** |
|------|--------|-----|-----|-----------------|
| **PushT image** | 0.66/0.47 | 0.70/0.66 | 0.69/0.60 | **0.91/0.84** |
| **Square** | 0.95/0.84 | 0.20/0.06 | 0.76/0.61 | **0.98/0.92** |
| **Can** | 0.99/0.94 | 0.00/0.00 | 0.82/0.72 | **1.00/0.97** |
| **Lift** | 1.00/0.98 | 0.04/0.00 | 0.97/0.94 | **1.00/1.00** |
| Transport | 0.66/0.53 | 0.00/0.00 | 0.68/0.53 | **1.00/0.93** |
| ToolHang | 0.53/0.26 | 0.00/0.00 | 0.15/0.03 | **0.95/0.73** |

**重点解读**:
- 跟 Table 1 同样的结论：DP-C 全面领先
- **PushT image 是唯一图像任务中 DP-C 不到 1.00 的**，说明图像任务更难
- **你的复现重点**：🟢 PushT image（验证视觉策略有效）

### Table 4 — 多阶段任务

| 任务 | BC-RNN | IBC | BET | **DP-C (CNN)** |
|------|--------|-----|-----|-----------------|
| BlockPush p1 | 1.00 | 0.00 | 0.96 | **0.36** |
| BlockPush p2 | 1.00 | 0.00 | 0.71 | **0.11** |
| Kitchen 平均 | 0.63/0.57 | 1.00/0.97 | 0.71/0.66 | **1.00/0.99** |

**重点解读**:
- DP-C 在 **BlockPush 上翻车了**（0.36/0.11），比 BC-RNN（1.00）差很多
- 论文解释：DP 需要精确的时序对齐，BlockPush 的多模态较强
- **你的复现重点**：❌ 不需要跑，结论已经写在论文里了

### 表格之间的关键发现

| 发现 | 证据 |
|------|------|
| DP-C 全面优于所有基线 | Table 1 & 2 所有行 |
| IBC 极不稳定 | Table 1 中 4/6 任务为 0.00 |
| BET 无法处理连续动作 | PushT 上 0.00 |
| CNN 优于 Transformer | Table 1 DP-C vs DP-T 大部分任务 |
| DP 在简单任务达 100% | Square/Can/Lift 满分 |
| DP 在多阶段任务较弱 | BlockPush 0.36 |

---

## 3. 核心实验（Priority 1 — 必须复现）

### PushT lowdim (Table 1)
轻量化（~10分钟）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 dataloader.batch_size=64 \
    training.rollout_every=5 training.checkpoint_every=5
```
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=0.95, avg=0.91 | A100 ~2h

### Square lowdim (Table 1)
轻量化（~15分钟）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 dataloader.batch_size=64 \
    training.rollout_every=5 training.checkpoint_every=5
```
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=1.00, avg=0.93 | A100 ~3h

### Can lowdim (Table 1)
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=1.00, avg=0.96

### Lift lowdim (Table 1)
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=1.00, avg=0.98

---

## 4. 扩展实验（Priority 2 — 建议复现）

### Transport lowdim (Table 1)
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    task.env_runner.n_envs=1
```
预期: max=0.94, avg=0.82 | 需32核CPU

### ToolHang lowdim (Table 1)
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=0.50, avg=0.30 | 最难的任务

---

## 5. 评估预训练模型

```bash
mkdir -p data
wget -P data/ https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt

python eval.py --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt \
    --output_dir data/pusht_eval_output --device cuda:0
cat data/pusht_eval_output/eval_log.json
```

---

## 6. 怎么看结果

```bash
# 训练完成后，看检查点文件名
ls data/outputs/2026.07.*/*/checkpoints/*.ckpt

# 算 max 和 avg(last10)
ls data/outputs/2026.07.*/*/checkpoints/epoch=*-test_mean_score=*.ckpt | \
    sed 's/.*test_mean_score=\([0-9.]*\).ckpt/\1/' | \
    awk '{a[++n]=$1; if($1>m)m=$1} END{print "Max="m; for(i=n-9;i<=n;i++){if(i>0){s+=a[i];c++}} print "Avg(last" c ")=" s/c}'
```

---

## 7. 复现优先级总结

| 优先级 | 实验 | 预期 | 说明 |
|--------|------|------|------|
| 🥇 | PushT lowdim | 0.95/0.91 | 最核心，验证 DP 有效 |
| 🥇 | Square lowdim | 1.00/0.93 | robomimic 标准任务 |
| 🥇 | Can lowdim | 1.00/0.96 | 同上 |
| 🥇 | Lift lowdim | 1.00/0.98 | 最简单的任务 |
| 🥈 | Transport lowdim | 0.94/0.82 | 双臂任务，看 scalability |
| 🥈 | ToolHang lowdim | 0.50/0.30 | 高精度，DP 的极限在哪 |
| 🥉 | PushT image | 0.91/0.84 | 验证图像策略 |

**核心验证点**：如果你的 PushT 达到 ~0.95，Square/Can/Lift 达到 0.95-1.00，你就成功复现了 Diffusion Policy 的核心结论。
