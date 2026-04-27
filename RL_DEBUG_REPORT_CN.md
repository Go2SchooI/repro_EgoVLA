# EgoVLA RL 后训练 Debug 记录

日期：2026-04-28

本文档整理 `open_laptop_smooth02` 这轮 EgoVLA RL 后训练中遇到的问题、已做的代码修复、训练指标解读、实验结果，以及后续建议。目标不是重新设计 RL 方案，而是把当前现象和判断依据记录清楚，方便后续继续调试。

## 1. 背景

本轮实现遵循 `RL_IMPLEMENTATION_SPEC.md` 中的方案：

- EgoVLA baseline 必须冻结，RL loss 不允许反传进 EgoVLA。
- Actor 插入点必须在 temporal smoothing 之后、IK/retarget/env.step 之前。
- `a_ref` 是原 EgoVLA pipeline smoothing 后本来要执行的 command。
- `a_exec` 是 RL actor postprocess 后实际传入 IK/env 的 command。
- Replay 中 `a_ref` 用于 actor condition 和 reference regularization，`a_exec` 用于 env transition 和 critic training。
- Reward 使用 sparse success：success 时 `reward=1`，否则 `0`。
- `done = success or timeout`。
- `rl.enabled=false` 时 baseline eval path 应保持不变。

本轮主要任务是基于 base replay 做 offline RL post-training，然后评估 RL actor 是否能提升 Open-Laptop 成功率。

## 2. 最初异常

最初 50000 step offline RL 输出非常异常：

```text
q_loss 从 1.9e3 增长到 1e11 量级
actor_loss 到 -2e7
q1_mean/q2_mean 到 1.9e7
log_prob 从 154 增长到 553
entropy 为负数
mean_abs_action_ref = 0
mean_abs_a_exec_minus_a_ref_norm = 0
```

初看像 SAC 严重发散。后续检查 replay 后确认有两个核心问题：

1. 旧 replay 的 action normalization 是 legacy z-score，`ref_action_norm` 最大值达到约 `9.24`。但 actor 是 tanh 输出，只能覆盖 `[-1, 1]`。这会让 actor 无法表达 reference action，也会让 reference loss 和 critic target 不稳定。
2. actor/critic observation 没有归一化。真实 replay 中曾观察到：

```text
actor_obs absmax 约 908
critic_obs absmax 约 1427
```

这会使 critic 很容易数值膨胀。

此外，早期指标名 `mean_abs_action_ref` 容易误导。base replay 是 collect_base 采样，因此数据中的 `a_exec == a_ref`，所以 dataset action 与 reference 的差为 0 是正常的，不代表 actor 没有输出偏移。

## 3. 已做代码修复

已做的关键修复包括：

- Action normalization 改为 `minmax_clip`，将 `a_ref` 映射到 tanh actor 可覆盖的区间。
- Offline RL 加入 replay fast fields 自动重建逻辑。旧 replay 会打印：

```text
offline_rl_rebuilding_fast_fields ... action_rebuild=True obs_rebuild=True
```

- 新增 actor/critic observation normalizer，并保存到 replay metadata 和 RL checkpoint。
- `eval_rl` 加载 checkpoint 时必须带 actor obs normalizer，否则报错。
- collect_base 结束后会同时拟合 action normalizer、actor obs normalizer、critic obs normalizer，并重建 fast fields。
- actor_obs 的末尾保留 `a_ref_norm` 原语义，不对这段做二次 z-score。
- `eval_rl` 必须显式加载 checkpoint，不再允许静默使用随机 actor。
- collect_base/debug_trace 不再对 `a_ref` 做 actor postprocess，保证 base replay 中 `a_exec == a_ref`。
- 去掉了一些不必要的硬编码检查，比如固定 `env.num_actions == 50` 和固定 actor dim assert。

## 4. 指标含义

### q_loss

代码逻辑近似为：

```python
target = reward + gamma * (1 - done) * (target_min_q - alpha * next_logp)
q1, q2 = critic(critic_obs, dataset_action)
q_loss = mse(q1, target) + mse(q2, target)
```

`q_loss` 表示 critic 对 replay 中数据动作的 Q 预测与 bootstrapped target 之间的均方误差。

注意：`q_loss` 低不等于任务成功率高。它只能说明 critic 在当前 Bellman target 上拟合得好，而这个 target 本身也依赖 actor 和 target critic。

### q1_mean / q2_mean

代码逻辑为：

```python
q1, q2 = critic(critic_obs, action)
q1_mean = q1.mean()
```

这里的 `action` 是 replay batch 中的 `a_exec_norm`。对于 collect_base replay，基本等于 `a_ref_norm`。

因此 `q1_mean` 表示 critic 对 replay 数据动作的平均价值估计，不是 actor 当前 deterministic action 的真实环境表现。

`q1_mean` 高不一定代表 actor 效果好。offline RL 中 critic 可能对 actor 产生的新动作过度乐观，也就是 Q overestimation 或 extrapolation error。

### log_prob / entropy

`log_prob` 是 tanh-squashed Gaussian actor 对采样 action 的 log probability，并对 action 维度求和。

连续分布的 log probability 可以为正，所以 `log_prob > 0` 不一定数学错误。但在本任务中，如果同时出现：

- `log_prob` 很快变成较大的正数；
- `entropy = -log_prob` 变成明显负数；
- `ref_loss` 反弹；
- actor residual 变大或策略 eval 失败；

则通常表示 policy 分布过尖，可能出现 policy collapse。

### mean_abs_actor_minus_ref_norm

该指标近似为：

```python
abs(pi_action - ref_action).mean()
```

其中：

- `pi_action` 是训练时 actor 采样得到的 normalized action；
- `ref_action` 是 `a_ref_norm`；
- 平均覆盖 batch 和全部 38 个 action 维度。

它表示 actor 输出相对 EgoVLA reference 的平均偏移。数值降低说明 actor 更接近 reference。

但它是全维平均，不能直接代表关键 EE 平移误差。对于 Open-Laptop 这种接触任务，即使平均 residual 不大，某几个 EE xyz 维度的偏移也可能让手指错过盖子边缘。

### ref_loss

该指标近似为：

```python
((pi_action - ref_action) ** 2).mean()
```

它是 normalized action space 中的 reference regularization loss。和 `mean_abs_actor_minus_ref_norm` 类似，都是衡量 actor 离 reference 多远，只是一个是 L2/MSE，一个是 L1/绝对值均值。

## 5. 已跑实验和现象

### 5.1 修复 normalization 后的 1000 step smoke

配置：

```text
update_steps=1000
lr=1e-4
beta_ref=5
alpha_entropy=0.2
```

现象：

- `log_prob` 为负，`entropy` 为正。
- `q_loss` 不再爆炸。
- `mean_abs_actor_minus_ref_norm` 从约 `0.69` 降到约 `0.44`。
- `q_mean` 到约 `16.7`，比最初爆炸小很多。

结论：normalization 修复有效，数值不再立即发散。

### 5.2 beta=5, alpha=0.05, 5000 steps

配置：

```text
update_steps=5000
lr=1e-4
beta_ref=5
alpha_entropy=0.05
```

现象：

- 前 2000 steps 健康。
- 3000 steps 后 `log_prob` 接近 0。
- 4000 steps 后 `log_prob` 变正，`entropy` 变负。
- 5000 steps 时 `mean_abs_actor_minus_ref_norm` 回升到约 `0.49`，`q_mean` 到约 `29.5`。

结论：后半段出现 offline SAC 过训练或 policy collapse，不建议用 5000 step checkpoint。

### 5.3 beta=5, alpha=0.05, 2000 steps

配置：

```text
update_steps=2000
lr=1e-4
beta_ref=5
alpha_entropy=0.05
```

训练现象：

- `q_loss` 低且稳定。
- `q1_mean/q2_mean` 到约 `4.2`。
- `log_prob` 保持负数。
- `mean_abs_actor_minus_ref_norm` 稳定在约 `0.30`。

训练指标看起来健康。

评估：

```text
Room 2 / Table 1 / episode_8 / trial 0
baseline collect_base: Result True, success=1.0
eval_rl_2k: Result False, success=0.0
```

根据视频观察，RL actor 不是完全停住，而是在 7 到 10 秒附近有开盖动作意图，但末端位置误差导致没有和盖子发生有效接触，之后末端逐渐回到初始位置。

结论：训练数值健康，但实际接触几何被 actor residual 破坏。

### 5.4 beta=20, alpha=0.05, 2000 steps

配置：

```text
update_steps=2000
lr=1e-4
beta_ref=20
alpha_entropy=0.05
```

现象：

- `mean_abs_actor_minus_ref_norm` 降到约 `0.167`。
- 但从 step 200 开始 `log_prob` 约 `+11`，`entropy` 约 `-11`。
- `q_mean` 变为负数并持续下降到约 `-4.25`。
- actor loss 为正且持续上升。

结论：reference regularization 太强，policy 分布很早变得过尖，不建议作为主要候选。

### 5.5 beta=10, alpha=0.05, lr=5e-5, 2000 steps

配置：

```text
update_steps=2000
lr=5e-5
beta_ref=10
alpha_entropy=0.05
```

训练现象：

- `q_loss` 稳定。
- `q_mean` 在约 `0.2` 到 `0.3` 附近，没有明显过估计飙升。
- `mean_abs_actor_minus_ref_norm` 稳定在约 `0.22`。
- `log_prob` 在 0 附近波动，最终约 `+0.69`，比 beta=20 好，但仍然偏尖。

评估：

```text
Room 2 / Table 1 / episode_8 / trial 0
eval_rl_beta10: Result False, success_rate=0
```

结论：虽然比 beta=5 保守，但 residual 仍可能足以破坏接触。单纯调 `beta_ref/lr/update_steps` 还没有解决问题。

## 6. 为什么指标变好但 eval 变差

当前现象的核心是：offline RL 训练指标衡量的是 replay 分布上的网络拟合和正则，不等价于真实仿真成功率。

具体原因：

1. `q_loss` 低只说明 critic 拟合当前 Bellman target，不说明真实任务成功。
2. `q1_mean` 是 critic 对 replay 数据动作的估计，不是实际 eval actor 的真实环境回报。
3. critic 没有真实评估 actor 新动作，在 offline RL 中容易对未见动作产生错误高估。
4. `mean_abs_actor_minus_ref_norm` 和 `ref_loss` 是全维平均，会掩盖关键 EE xyz 上的小偏移。
5. Open-Laptop 是精确接触任务，末端位置轻微偏移就可能导致错过盖子边缘。
6. Eval 使用 deterministic mean，而训练指标里的 actor action 是 sampled action，两者不完全一致。

因此，当前更像是 actor 学到了一个看起来不大的 residual，但该 residual 破坏了关键接触几何。

## 7. 当前判断

当前不是 action path 或 checkpoint 加载错误：

- `rl_loaded_checkpoint` 正常出现。
- normalizer 正常加载。
- eval rollout 能跑满 300 step。
- 失败是 timeout，不是程序异常。
- 同一场景 baseline 曾成功，说明任务本身不是不可解。

当前主要问题是策略行为层面：

- actor residual 对精确接触任务仍然太大；
- 全维 average regularization 不能保证 EE 接触点正确；
- vanilla offline SAC 在 sparse reward 和小 replay 上容易把策略推向 critic 认为好的但环境中无效的动作。

## 8. 建议的下一步

### 8.1 先验证当前环境 baseline

由于后续 eval 日志中出现过远程 asset 加载失败和材质缺失 warning，建议在当前环境状态下重新跑同一点 baseline：

```bash
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
  Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
  rl_runs/open_laptop_smooth02/eval_base_now_result.txt \
  0 0 0.2 \
  rl_runs/open_laptop_smooth02/eval_base_now_videos eval_base_now 0
```

如果 baseline 仍成功，则可以确认 RL actor 破坏了接触。若 baseline 当前也失败，需要先排除环境和 asset 加载变化的影响。

### 8.2 不建议继续只靠增大 beta_ref

`beta=20` 已经出现明显 policy collapse，`beta=10` 仍 eval 失败。继续简单增大 beta_ref 可能只会让 actor 变僵硬或退化成 baseline。

### 8.3 更推荐 eval-time residual scale

当前最直接的思路是保留 actor 学到的方向，但缩小执行 residual：

```python
a_exec_norm = a_ref_norm + scale * (actor_norm - a_ref_norm)
```

建议先试：

```text
scale = 0.25
scale = 0.5
```

这可以验证“actor 方向有用但幅度太大”这个假设。若 scale 后成功率恢复或提升，说明问题主要是 residual magnitude，而不是 actor 完全学错。

### 8.4 增加更细粒度日志

建议后续记录：

- left/right EE xyz residual；
- left/right EE quaternion residual；
- left/right hand residual；
- deterministic actor residual，而不是只看 sampled actor residual；
- per-step `a_ref`、`a_exec`、`a_exec - a_ref`；
- 成功/失败视频对应的 residual 曲线。

对于 Open-Laptop，特别需要观察右手/左手末端是否在开盖关键阶段偏离盖子边缘。

## 9. 重要文件和视频路径

baseline 成功视频：

```text
rl_runs/open_laptop_smooth02/videos/collect_base/inference_0.2_0.2/Open-Laptop/room_2/table_1/Open-Laptop_room_2_table_1_episode_('episode_8.hdf5', 0.008424)_0.mp4
```

beta=5, 2k eval 失败视频：

```text
rl_runs/open_laptop_smooth02/eval_rl_2k_videos/eval_rl_2k_alpha005/inference_0.2_0.2/Open-Laptop/room_2/table_1/Open-Laptop_room_2_table_1_episode_('episode_8.hdf5', 0.008424)_0.mp4
```

beta=10 eval 失败视频：

```text
rl_runs/open_laptop_smooth02/eval_rl_beta10_videos/eval_rl_beta10_alpha005_lr5e5/inference_0.2_0.2/Open-Laptop/room_2/table_1/Open-Laptop_room_2_table_1_episode_('episode_8.hdf5', 0.008424)_0.mp4
```

主要 checkpoint：

```text
rl_runs/open_laptop_smooth02/rl_checkpoint_2k_alpha005.pt
rl_runs/open_laptop_smooth02/rl_checkpoint_2k_beta20_alpha005.pt
rl_runs/open_laptop_smooth02/rl_checkpoint_1200_beta10_alpha005_lr5e5.pt
```

注意：`rl_checkpoint_1200_beta10_alpha005_lr5e5.pt` 文件名里有 `1200`，但实际命令中 `--rl_update_steps 2000`，因此实际训练步数是 2000。

