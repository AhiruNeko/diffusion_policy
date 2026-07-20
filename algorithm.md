# Diffusion Policy 算法原理

## 核心思想

策略建模为**条件去噪扩散过程**：给定观测 o，从高斯噪声逐步去噪得到动作序列。

## 训练 (compute_loss)

```python
# policy/diffusion_unet_lowdim_policy.py :183-252
obs → normalize → global_cond       # 观测作为条件
trajectory = actions                 # 动作序列
noise = 采样高斯噪声                  # ε
noisy = 加噪(trajectory, t)          # √ᾱ·x + √(1-ᾱ)·ε
pred = UNet(noisy, t, global_cond)   # 预测 ε
loss = MSE(pred, noise)              # 简单 MSE 损失
```

## 推理 (predict_action)

```python
# policy/diffusion_unet_lowdim_policy.py :99-177
x_T = 采样高斯噪声                    # 从纯噪声开始
for t = T → 0:
    pred_noise = UNet(x_t, t, cond)   # 预测噪声
    x_{t-1} = scheduler.step(...)     # 一步去噪
x_0 = 去噪完成的动作序列                # 最终动作
执行前 n_action_steps 步                # 滑窗控制
```

## 关键组件

### Conditional UNet 1D
```
输入: (B, T, Da)                    # 动作序列
 ↓ Conv1D + SiLU
ResBlock × 3 (下采样)                # FiLM 调制注入观测条件
 ↓
ResBlock × 2 (中间)
 ↓
ResBlock × 3 (上采样) + skip connections
 ↓
输出: (B, T, Da)                    # 预测的噪声
```

### 条件注入 (global_cond)
```
obs (B, To, Do) → flatten → (B, To·Do) → concat([time_embed, cond])
→ 每一层 ResBlock 做 FiLM: gamma·x + beta
```

### Receding Horizon
预测 horizon 步动作，只执行前 n_action_steps 步，然后重新规划。

### Inpainting Mask
训练时部分观测位置填入真实值，推理时保持观测位置不变。

## 为什么有效

- **多模态**: 不同噪声种子 → 不同轨迹，表达多模态动作分布
- **时序一致**: 同时预测多步动作，避免单步策略的抖动
- **训练稳定**: MSE 损失比对抗训练稳定得多
