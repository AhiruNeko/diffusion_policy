# 复现速查清单

## 实验优先级

| 优先级 | 任务 | 命令 | 预期 | 时间 |
|--------|------|------|------|------|
| 🥇1 | PushT lowdim | `train_diffusion_unet_lowdim_workspace.yaml` | 0.95/0.91 | ~2h A100 |
| 🥇2 | Square lowdim | `task=square_lowdim` | 1.00/0.93 | ~3h |
| 🥇3 | Can lowdim | `task=can_lowdim` | 1.00/0.96 | ~3h |
| 🥇4 | Lift lowdim | `task=lift_lowdim` | 1.00/0.98 | ~3h |
| 🥇5 | PushT image | `image_pusht_diffusion_policy_cnn.yaml` | 0.91/0.84 | ~6h |

## 代码重点

| 文件 | 内容 |
|------|------|
| `policy/diffusion_unet_lowdim_policy.py` :183-252 | compute_loss() |
| `policy/diffusion_unet_lowdim_policy.py` :99-177 | predict_action() |
| `policy/diffusion_unet_lowdim_policy.py` :59-96 | conditional_sample() |
| `model/diffusion/conditional_unet1d.py` :173-241 | UNet forward() |

## 老师高频问题

- **DP 比 BC-RNN 好在哪？** 多模态 + 时序一致
- **同时预测多步作用？** 避免动作在模态间切换
- **观测怎么传入？** global_cond 展平后 FiLM 调制
- **0.95 怎么算的？** 50 episode 最大平均，3 种子平均
