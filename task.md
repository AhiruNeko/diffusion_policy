# Diffusion Policy 论文复现指南

**论文**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2303.04137v5)

---

## 1. 前置条件

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
cd ~/projects/diffusion_policy
```

## 2. 核心实验（Priority 1）

### PushT lowdim (Table 1)
轻量化（RTX 4060 8GB，~10分钟）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 training.batch_size=64
```
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=0.95, avg=0.91 | A100 ~2h | RTX4060 ~8h

### Square lowdim (Table 1)
轻量化（RTX 4060 8GB，~15分钟）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 training.batch_size=64
```
完整训练：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=1.00, avg=0.93 | A100 ~3h | RTX4060 ~12h (batch_size=128)

### Can lowdim (Table 1)
轻量化：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=can_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 training.batch_size=64
```
预期: max=1.00, avg=0.96

### Lift lowdim (Table 1)
轻量化（最轻，~5分钟）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=lift_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 training.batch_size=64
```
预期: max=1.00, avg=0.98

## 3. 扩展实验（Priority 2）

### Transport lowdim (Table 1)
轻量化（n_envs=1 减少CPU负载）：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=10 training.batch_size=32 task.env_runner.n_envs=1
```
预期: max=0.94, avg=0.82 | 需32核CPU，建议服务器

### ToolHang lowdim (Table 1)
轻量化：
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20 training.batch_size=64
```
预期: max=0.50, avg=0.30

## 4. PushT image (Table 2)
轻量化（减小batch_size避免OOM）：
```bash
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=10 training.batch_size=16
```
完整训练：
```bash
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled
```
预期: max=0.91, avg=0.84 | A100 ~6h | 建议服务器

## 5. 评估预训练模型
```bash
mkdir -p data
wget -P data/ https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt

python eval.py --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt \
    --output_dir data/pusht_eval_output --device cuda:0
cat data/pusht_eval_output/eval_log.json
```

## 6. 轻量化验证（本地 10 分钟）
```bash
python train.py --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    training.seed=42 training.device=cuda:0 logging.mode=disabled \
    training.num_epochs=20
```

## 7. 你的 GPU 能跑什么

| 实验 | RTX 4060 8GB | 建议 |
|------|-------------|------|
| PushT lowdim | ✅ 能跑 | batch_size=256, ~8h |
| Square/Can/Lift lowdim | ✅ 能跑 | ~12h each |
| Transport lowdim | ⚠️ CPU 瓶颈 | 建议服务器 |
| ToolHang lowdim | ✅ 能跑 | ~8h |
| PushT image | ⚠️ 可能 OOM | 建议服务器 |
| IBC | ✅ 能跑 | ~4h |
| BET | ✅ 能跑 | ~4h |
| BC-RNN | ✅ 能跑 | ~6h |
