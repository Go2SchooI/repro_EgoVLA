# EgoVLA RL 诊断实验结果总结

日期：2026-04-28

本文档总结 `open_laptop_smooth02` 这轮诊断实验结果。目标是判断 RL eval 失败更像是实现路径问题、任务敏感性问题，还是 actor residual 幅度/方向问题。本文不讨论重新设计 RL，不新增 reward，也不把 offline 训练指标等同于真实 eval 成功。

## 1. 实验对象

任务设置：

```text
task: Humanoid-Open-Laptop-v0
room_idx: 2
table_idx: 1
smooth_weight: 0.2
hand_smooth_weight: 0.2
num_episodes: 1
num_trials: 1
episode: episode_8.hdf5
trial: 0
```

主要 checkpoint：

```text
rl_runs/open_laptop_smooth02/rl_checkpoint_1200_beta10_alpha005_lr5e5.pt
```

主要诊断输出：

```text
rl_runs/open_laptop_smooth02/diag_baseline_result.txt
rl_runs/open_laptop_smooth02/diag_identity_result.txt
rl_runs/open_laptop_smooth02/diag_noise_*_result.txt
rl_runs/open_laptop_smooth02/diag_beta10_scale_*_result.txt
rl_runs/open_laptop_smooth02/diag_*_action_diff.csv
rl_runs/open_laptop_smooth02/diag_videos/
```

## 2. 总体结论

当前最可信的阶段性结论是：

```text
baseline 和 identity 都成功，说明 RL 插入路径没有明显破坏 baseline。
tiny-noise 0.01 / 0.03 / 0.05 都成功，说明该单点不是极小扰动即失败的极端敏感状态。
residual scale 0.0 / 0.1 / 0.25 / 0.5 都成功，1.0 失败，说明 actor residual 方向不是完全错误，更像 full scale 幅度过大。
```

因此，当前失败不应优先解释为：

- pack/unpack 大错位；
- normalizer/checkpoint load 完全坏掉；
- Open-Laptop 对任何微小扰动都立刻失败；
- actor residual 方向完全不可用。

更合理的解释是：

```text
当前 beta10 checkpoint 的 actor residual 在小到中等 scale 下可接受，但 full scale 过强，尤其 hand group 和 right_ee_rot 在关键阶段被明显改动，导致最终 laptop 打开幅度不足。
```

## 3. Baseline 与 Identity

结果：

| method | result | success | move_lid_success | video duration |
|---|---:|---:|---:|---:|
| baseline | True | 1.0 | 1.0 | 6.2s |
| identity actor | True | 1.0 | 1.0 | 6.2s |

结果文件：

```text
rl_runs/open_laptop_smooth02/diag_baseline_result.txt
rl_runs/open_laptop_smooth02/diag_identity_result.txt
```

identity action diff：

```text
rows: 564
steps: 94
groups per step: 6
```

identity 并不是数学上完全 `a_exec == a_ref`，但任务仍成功。主要非零误差来自 `postprocess_action`：

| group | mean(mean_abs) | max(max_abs) | 解释 |
|---|---:|---:|---|
| left_ee_pos | 0.0 | 0.0 | 完全一致 |
| right_ee_pos | 5.28e-11 | 1.49e-08 | 浮点误差 |
| left_ee_rot | 1.03e-04 | 0.00121 | quaternion normalize |
| right_ee_rot | 0.00125 | 0.02684 | quaternion normalize |
| left_hand | 1.83e-09 | 3.73e-08 | 浮点误差 |
| right_hand | 0.00116 | 0.11268 | hand DOF clamp |

最明显的 identity 差异在 right hand：

```text
step 6
global dim 34
ref  = -0.212680
exec = -0.100000
diff = +0.112680
```

这说明 identity path 经过了 hand limit clamp。由于 identity 仍成功，这个 clamp 本身在当前单点不是失败原因。但后续解释 lambda=0 时要注意：lambda=0 等价于 RL identity path，而不是完全原始 baseline path。

## 4. Tiny-Noise Sensitivity

实验：在 normalized action space 对 `a_ref_norm` 加 Gaussian noise：

```text
a_exec_norm = clamp(a_ref_norm + noise_scale * epsilon)
noise_seed = 123
```

结果：

| noise_scale | result | success | move_lid_success | steps | video duration |
|---:|---:|---:|---:|---:|---:|
| 0.01 | True | 1.0 | 1.0 | 94 | 6.2s |
| 0.03 | True | 1.0 | 1.0 | 94 | 6.2s |
| 0.05 | True | 1.0 | 1.0 | 95 | 6.267s |

per-group raw diff 概览：

| noise_scale | top groups |
|---:|---|
| 0.01 | right_hand mean 0.00482 max 0.11453；right_ee_rot mean 0.00458 max 0.02895 |
| 0.03 | right_hand mean 0.01231 max 0.11518；right_ee_rot mean 0.01228 max 0.04818 |
| 0.05 | right_ee_rot mean 0.02037 max 0.08133；right_hand mean 0.01969 max 0.12898 |

解释：

```text
tiny-noise 到 0.05 仍成功，说明当前单点不是“normalized action 里极小扰动就失败”的状态。
```

这会削弱“Open-Laptop 极端敏感，任何 residual 都不行”的解释。

## 5. Residual Scale Sweep

实验公式：

```text
a_exec_norm = a_ref_norm + lambda * (a_actor_norm - a_ref_norm)
```

结果：

| lambda | result | success | move_lid_success | steps | video duration |
|---:|---:|---:|---:|---:|---:|
| 0.0 | True | 1.0 | 1.0 | 94 | 6.2s |
| 0.1 | True | 1.0 | 1.0 | 94 | 6.2s |
| 0.25 | True | 1.0 | 1.0 | 95 | 6.267s |
| 0.5 | True | 1.0 | 1.0 | 99 | 6.534s |
| 1.0 | False | 0.0 | 1.0 | 300 | 20.0s |

Open-Laptop 的 success 定义：

```text
move_lid_success: laptop joint > upper_limit * 0.15
success:          laptop joint > upper_limit * 0.70
```

因此 `lambda=1.0` 的失败不是完全没开盖，而是只达到 `move_lid_success=1.0`，没有达到最终 `success=1.0` 的 70% 开盖阈值。

## 6. Residual Group 分析

为了公平比较，`lambda=1.0` 只统计前 94 step，即和成功 run 接近的时间窗口。否则 1.0 跑满 300 step 后的失败分布漂移会污染均值。

前 94 step 内主要 group raw diff：

| lambda | right_hand mean/max | left_hand mean/max | right_ee_rot mean/max | right_ee_pos mean/max |
|---:|---:|---:|---:|---:|
| 0.0 | 0.00111 / 0.11261 | 0.00000 / 0.00000 | 0.00125 / 0.02706 | 0.00000 / 0.00000 |
| 0.1 | 0.00351 / 0.11532 | 0.00160 / 0.01160 | 0.00249 / 0.02221 | 0.00098 / 0.00416 |
| 0.25 | 0.00698 / 0.11317 | 0.00452 / 0.02910 | 0.00508 / 0.02219 | 0.00227 / 0.00983 |
| 0.5 | 0.01350 / 0.10616 | 0.01001 / 0.05796 | 0.00939 / 0.03845 | 0.00434 / 0.01897 |
| 1.0 | 0.03028 / 0.22989 | 0.02344 / 0.11568 | 0.01915 / 0.07182 | 0.00793 / 0.03602 |

`lambda=1.0` 在早期 step 2-6 的 right_hand residual 明显偏大：

```text
step 3 right_hand:
mean_abs = 0.08481
l2       = 0.41814
max_abs  = 0.22989

主要维度：
global dim 32: diff -0.22989
global dim 29: diff -0.20408
global dim 33: diff -0.19544
global dim 28: diff -0.17153
```

`lambda=0.5` 对应早期最大 right_hand residual 明显较小：

```text
scale 0.5 step 6 right_hand max_abs = 0.10616
scale 1.0 step 3 right_hand max_abs = 0.22989
```

失败后的后段还出现很大的 right_ee_rot 漂移，例如：

```text
lambda=1.0 step 130 right_ee_rot max_abs = 0.82782
```

但这更像失败后闭环状态分布漂移，不应单独当作初始失败原因。更应该关注前 94 step 的 hand 和 right_ee_rot residual 逐步放大。

## 7. 对诊断假设的判断

| 假设 | 当前判断 | 依据 |
|---|---|---|
| H1: RL 插入路径 / pack-unpack / normalizer 有大 bug | 暂不支持 | baseline 成功，identity 成功，lambda=0 成功 |
| H2: Open-Laptop 对 tiny perturbation 极端敏感 | 暂不支持 | noise 0.01 / 0.03 / 0.05 全成功 |
| H3: actor residual 方向可能有用但幅度过大 | 支持 | lambda 0.1 / 0.25 / 0.5 成功，lambda 1.0 失败 |
| H4: actor residual 方向本身错误 | 不强支持 | 非零小 scale 成功，但 full scale 某些 group 仍可能方向/幅度不佳 |
| H5: offline critic 对 actor 新动作 OOD overestimation | 尚未验证 | 当前诊断没有记录 `Q(s,a_ref) / Q(s,a_actor) / Q(s,a_scaled)` |
| H6: 单点 eval 不可靠，需要 paired 统计 | 仍成立 | 当前结果只覆盖一个 room/table/episode/trial |

## 8. 当前建议

短期不建议继续盲目增加训练步数或单纯调大 `beta_ref`。当前更值得做的是：

1. 细化 residual scale 阈值：

```text
lambda = 0.6, 0.7, 0.8, 0.9
```

目标是确认当前 checkpoint 的 full actor residual 可用上限是否在 `0.5-0.8` 之间。

2. 做 paired 多 trial 统计：

```text
method: baseline, identity, tiny_noise_0.05, scale_0.5, scale_1.0
N: 至少 10，最好 20
固定 task / room / table / episode / trial / randomize_idx 生成规则
```

3. 如果后续做代码诊断增强，优先补：

```text
summary csv: success, timeout, episode_len, video_path, randomize_idx
normalized residual stats: a_ref_norm, a_exec_norm, a_actor_norm
critic OOD stats: Q(s,a_ref), Q(s,a_actor), Q(s,a_scaled)
checkpoint hash/update_step logging
```

4. 如果后续要做策略限制，优先关注：

```text
right_hand group
left_hand group
right_ee_rot group
```

但这属于后续方案讨论，不是本轮诊断接口审查的范围。

## 9. 一句话总结

当前结果说明：RL 插入路径基本可信，任务不是 tiny-noise 即崩，beta10 actor 的 residual 小 scale 可用，full scale 过强；`lambda=1.0` 失败更像 actor residual 幅度或局部 group 改动过大导致开盖幅度不足，而不是完全无效策略。
