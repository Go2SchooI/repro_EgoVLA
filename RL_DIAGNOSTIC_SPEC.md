# EgoVLA RL 后训练诊断规格

日期：2026-04-28

目标：把“RL actor 为什么让 Open-Laptop 失败”拆成可验证诊断树。本文不主张继续调 SAC、换 reward、改网络或盲目训练；先用 baseline / identity / tiny-noise / residual-scale / per-group logging 区分实现 bug、任务敏感性、residual 幅度、residual 方向、offline critic OOD、单点 eval 噪声。

## 1. 当前失败现象

基于 `RL_DEBUG_REPORT_CN.md` 和 `rl_runs/open_laptop_smooth02`：

- 同一 Open-Laptop 设置中，baseline 曾成功，RL checkpoint eval 失败。
- normalization bug 修复后，训练数值不再明显发散，`q_loss` 可低且稳定。
- beta=5 2k、beta=10 2k 类 checkpoint 训练指标看起来可接受，但单点 eval 仍失败。
- 视频观察显示 RL actor 并非完全停住，而是有开盖意图，但关键接触几何被破坏。
- `collect_result.txt` 也显示 Open-Laptop 在不同 room/table/episode 下有成功有失败，单个视频不能代表统计结论。

## 2. 为什么不能完全相信之前判断

之前“主要是 residual 太大/策略问题”的判断有一定合理性，但还不能作为结论：

- `q_loss` 低只说明 critic 拟合 Bellman target，不说明真实环境成功。
- `q1_mean/q2_mean` 目前主要反映 replay 数据动作，不等于 actor 新动作的真实回报。
- `mean_abs_actor_minus_ref_norm` 是 38D 全维平均，可能掩盖关键 EE xyz/rot 的小偏差。
- 当前还没有 identity actor paired eval，不能排除 pack/unpack、normalizer、postprocess 或 checkpoint eval path 的隐藏 bug。
- Open-Laptop 可能对极小 action perturbation 本来就敏感。
- 单个 room/table/episode/trial 的成功或失败都可能是 paired eval 噪声。

## 3. 静态 Action Path 核查表

| 检查项 | 当前代码位置 | 结论 | 证据 | 风险 |
|---|---|---|---|---|
| `rl.enabled=false` 是否走原始 EgoVLA path | `ik_agent_30hz.py` eval loop | 静态看是 | `rl_enabled` 为 false 时 `ik_eval_single_step` 不传 `return_rl_features`，不进入 pack/actor/unpack 分支，直接 `smooth_action -> ik_step` | 仍需 paired baseline 确认同一环境当前可复现成功 |
| `rl.enabled=true` identity 是否复现 baseline | 新增 `rl.mode=eval_identity_actor` | 需 runtime 验证 | 该 mode 强制 `a_exec_norm = normalize(a_ref)`，再 denorm/postprocess/unpack/IK | normalizer clip、quat postprocess 可能造成非零误差，必须看 identity error |
| `a_ref` 是否为 smoothing 后 command | `ik_agent_30hz.py` smoothing 后 `pack_action` | 静态确认 | `action_left_ee/right_ee/hand` 先由 `smooth_action` 得到，再 `pack_action` | 若 smoothing 队列来自 base chunk 而执行动作来自 actor，会有闭环分布偏差，这是设计选择 |
| `a_exec` 是否实际送入 `ik_step` | `ik_agent_30hz.py` actor branch 后 `unpack_action` | 静态确认 | `a_exec -> unpack_action -> action_left/right_ee/hand -> ik_step -> env.step(action)` | `postprocess_action` 会 normalize quat、clamp hand DOF，日志需记录 postprocess 后结果 |
| critic 是否使用 `a_exec` | `ik_agent_30hz.py::_rl_append_transition`, `features.py::make_fast_fields`, `sac.py::update` | 静态确认 | raw field 存 `a_exec`；fast `action = normalizer.normalize(raw_fields["a_exec"])`；critic 用 batch `action` | online terminal transition 目前用 same ctx 作为 next；done mask 应屏蔽 target |
| reference loss 是否用 normalized actor/ref | `sac.py::update` | 静态确认 | `ref_loss_per = (pi_action - ref_action)^2`，两者来自 normalized action space | actor loss 用 sampled `pi_action`，eval 用 deterministic `tanh(mu)`，两者需分开解释 |
| pack/unpack slice 顺序 | `rl/action_space.py` | 静态确认 | `left_ee[0:7] right_ee[7:14] left_hand[14:26] right_hand[26:38]` | per-group logging 按同一 layout 分组 |
| normalizer 是否同一套 | `features.py`, `ik_agent_30hz.py` | 静态确认 | replay fast action/ref 和 eval actor denorm 都使用 `rl_action_normalizer` | identity mode 若未提供 replay/action normalizer，会退化为 identity normalizer |
| deterministic eval 是否用 `tanh(mu)` | `actor_critic.py::GaussianActor.sample`, `ik_agent_30hz.py` | 静态确认 | deterministic 分支 `u=mu; action=tanh(u)`；`eval_rl`/scale mode 使用 `rl_deterministic_eval` | 训练指标里的 sampled action 不能直接等价 eval 动作 |
| checkpoint 是否加载到 eval actor | `ik_agent_30hz.py`, `sac.py::load` | 静态确认 + runtime log | `eval_rl` 和 `eval_residual_scale_sweep` 要求 `--rl_load_rl_checkpoint_path`，加载后打印 `rl_loaded_checkpoint` | 必须检查 stdout 和 checkpoint path；不要允许随机 actor 静默 eval |
| `a_exec` 是否被后续 smoothing/overwrite | `ik_agent_30hz.py` actor branch 后 | 静态确认无后续 smoothing | actor branch 后只 `unpack -> ik_step`，没有再次 smooth | `ik_step` 会通过 IK 和 env joint limit 改写最终 50D env action，这是预期 |

不能静态确认的项必须通过 `eval_identity_actor`、`debug_trace_action_path` 和 action diff logs 验证。

## 4. 诊断树

### H1: RL 插入/pack/normalizer/checkpoint 仍有 bug

实验：

- baseline: `rl.enabled=false`
- identity actor: `rl.mode=eval_identity_actor`
- debug trace: `rl.mode=debug_trace_action_path`

判据：

- baseline 成功、identity 失败：停止 RL 分析，先修插入路径。
- identity 的 `identity_max_abs_error` 显著非零：检查 normalizer clip、postprocess、pack/unpack。
- `eval_rl` 没有打印 `rl_loaded_checkpoint`：停止，先修 checkpoint load。

### H2: Open-Laptop 对 action perturbation 极端敏感

实验：

- `rl.mode=eval_tiny_noise`
- `noise_scale = 0.01, 0.03, 0.05, 0.1`
- 固定 `noise_seed`，使用同一 room/table/episode/trial。

判据：

- 0.01 或 0.03 normalized noise 即明显破坏 baseline 成功：说明任务/控制链极敏感，后续 actor 必须更保守，可能需要只改部分维度或先换更鲁棒任务验证。

### H3: actor residual 方向有用但幅度过大

实验：

- `rl.mode=eval_residual_scale_sweep`
- 对同一 checkpoint 测 `lambda = 0.0, 0.1, 0.25, 0.5, 1.0`
- 执行动作：

```text
a_exec_norm = a_ref_norm + lambda * (a_actor_norm - a_ref_norm)
```

判据：

- `lambda=0` 成功、`lambda=0.1/0.25` 成功、`lambda=1` 失败：方向可能有用但幅度过大。
- `lambda=0` 成功、所有非零 lambda 都失败：方向可能错误，或任务极敏感。
- `lambda=0` 失败：先回 identity/baseline debug。

### H4: actor residual 方向本身错误

实验：

- residual scale sweep + per-group residual 曲线。
- 重点看 Open-Laptop 接触关键窗口中的 EE position/rotation。

判据：

- 小 lambda 就把接触阶段的 EE pos/rot 推离 baseline，且 tiny-noise 不那么敏感：方向本身更可疑。
- actor residual 主要集中在错误手、错误 EE rot 或手 DOF：优先限制对应 group。

### H5: offline critic 对 actor 新动作 OOD overestimation

实验：

- 比较 replay data action Q、actor action Q、scaled residual action Q。
- 当前本文先不改 SAC，只把该诊断列为下一步最小接口：在 offline eval/analysis 中记录 `Q(s,a_ref)`, `Q(s,a_actor)`, `Q(s,a_scaled)`。

判据：

- critic 给 actor/scaled OOD 动作更高 Q，但环境 success 下降：offline critic OOD/overestimation 可能成立。

### H6: 单点 eval 结论不可靠

实验：

- paired eval，至少 N=10 或 N=20 trial/episode 对。
- 同一 task、room/table、episode list、trial list、randomize_idx 生成规则。

判据：

- 单点失败但 paired 成功率差异不显著：不要过拟合单个视频。
- paired 中 identity 与 baseline 不一致：优先修代码。

## 5. 已新增最小诊断接口

本轮只加 eval/debug 接口和日志，不改 SAC loss、actor/critic architecture、reward、online RL。

新增 `rl.mode`：

- `eval_identity_actor`
- `eval_tiny_noise`
- `eval_residual_scale_sweep`

新增参数：

```text
--rl_noise_scale
--rl_noise_type gaussian|uniform
--rl_noise_seed
--rl_residual_scale
--rl_action_diff_log_path
--rl_action_diff_step_start
--rl_action_diff_step_end
```

新增日志工具：

- `human_plan/ego_bench_eval/rl/diagnostics.py`
- CSV: `per_step_action_diff.csv`
- PT: 同路径 `.pt`

per-group layout：

```text
left_ee_pos:  [0:3]
left_ee_rot:  [3:7]
right_ee_pos: [7:10]
right_ee_rot: [10:14]
left_hand:    [14:26]
right_hand:   [26:38]
```

CSV 字段包括：

```text
step, group_name, method, episode, trial, room_idx, table_idx,
checkpoint, lambda, noise_scale,
mean_abs, l2, max_abs,
a_ref_slice, a_exec_slice, a_actor_slice,
actor_minus_ref_mean_abs, actor_minus_ref_l2, actor_minus_ref_max_abs
```

runtime logs：

- `identity_actor_step identity_max_abs_error ... per_group_identity_error ...`
- `identity_actor_summary identity_max_abs_error ...`
- `rl_action_diff_summary mode=... summary=...`
- `rl_action_diff_saved path=...`

## 6. Mode 说明

### baseline

配置：

```text
rl.enabled=false
```

判据：

- 用当前环境复验 baseline 是否仍成功。
- 如果 baseline 当前也失败，不要分析 RL actor，先排除环境/asset/seed 变化。

### eval_identity_actor

执行：

```text
a_ref_norm = normalize(a_ref)
a_exec = postprocess(denormalize(a_ref_norm), ref=a_ref)
```

说明：

- 必须经过 pack/unpack、normalizer roundtrip、postprocess 和原 `ik_step`。
- 如果没有提供 replay/action normalizer，则使用 identity normalizer，误差应接近 0。
- 如果使用 minmax normalizer，边界处可能出现约 `0.001 * std` 的 clip roundtrip 误差；这正是该 mode 要暴露的风险。

必须检查：

```text
identity_max_abs_error
identity_mean_abs_error
per_group_identity_error
```

### eval_tiny_noise

执行：

```text
a_exec_norm = clamp(a_ref_norm + noise_scale * epsilon)
```

参数：

```text
--rl_noise_scale 0.01|0.03|0.05|0.1
--rl_noise_type gaussian|uniform
--rl_noise_seed <int>
```

判据：

- 小 noise 也失败：任务/控制链敏感。
- 小 noise 成功、actor 失败：actor residual 更可疑。

### eval_residual_scale_sweep

当前代码用单个 `--rl_residual_scale` 运行一次；paired runner 用 shell loop 形成 sweep。

执行：

```text
a_actor_norm = actor(actor_obs, deterministic=True)
a_exec_norm = clamp(a_ref_norm + lambda * (a_actor_norm - a_ref_norm))
```

参数：

```text
--rl_residual_scale 0.0|0.1|0.25|0.5|1.0
--rl_load_rl_checkpoint_path <checkpoint>
```

判据见 H3/H4。

## 7. Paired Eval Protocol

同一批 comparison 必须固定：

- task: first version `Humanoid-Open-Laptop-v0`
- room/table: first point `room_idx=2`, `table_idx=1`
- episode list: first version `num_episodes=1` 即 `episode_8.hdf5`
- trial list: first version `num_trials=1`，然后扩展 N=10/N=20
- smoothing: `smooth_weight=0.2`, `hand_smooth_weight=0.2`
- checkpoint/replay/action normalizer paths
- eval device and attention backend

建议 summary csv schema：

```text
task, room_idx, table_idx, episode, trial, method,
noise_scale, lambda, checkpoint,
success, timeout, episode_len, result_video_path,
mean_abs_residual_norm, max_abs_residual_norm,
group_residual_stats
```

第一版可以由每个 run 的 result txt + per-step action diff csv 聚合生成。不要只看视频。

## 8. 命令模板

公共前缀：

```bash
cd /root/gpufree-data/EgoVLA_Release
export EGO_VLA_EVAL_DEVICE=cuda:0
export EGO_VLA_SIGLIP_ATTN_IMPLEMENTATION=sdpa
```

### 8.1 baseline 复验

```bash
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
  Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
  rl_runs/open_laptop_smooth02/diag_baseline_result.txt \
  0 0 0.2 \
  rl_runs/open_laptop_smooth02/diag_videos baseline_now 0
```

### 8.2 identity actor

```bash
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
  Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
  rl_runs/open_laptop_smooth02/diag_identity_result.txt \
  0 0 0.2 \
  rl_runs/open_laptop_smooth02/diag_videos identity_actor 0 \
  --rl_enabled true \
  --rl_mode eval_identity_actor \
  --rl_replay_path rl_runs/open_laptop_smooth02/base_replay.pt \
  --rl_action_diff_log_path rl_runs/open_laptop_smooth02/diag_identity_action_diff.csv
```

### 8.3 tiny-noise sensitivity

```bash
for scale in 0.01 0.03 0.05 0.1; do
  bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
    Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
    "rl_runs/open_laptop_smooth02/diag_noise_${scale}_result.txt" \
    0 0 0.2 \
    rl_runs/open_laptop_smooth02/diag_videos "tiny_noise_${scale}" 0 \
    --rl_enabled true \
    --rl_mode eval_tiny_noise \
    --rl_replay_path rl_runs/open_laptop_smooth02/base_replay.pt \
    --rl_noise_scale "$scale" \
    --rl_noise_type gaussian \
    --rl_noise_seed 123 \
    --rl_action_diff_log_path "rl_runs/open_laptop_smooth02/diag_noise_${scale}_action_diff.csv"
done
```

### 8.4 residual scale sweep

```bash
CKPT=rl_runs/open_laptop_smooth02/rl_checkpoint_2k_alpha005.pt
for lam in 0.0 0.1 0.25 0.5 1.0; do
  bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
    Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
    "rl_runs/open_laptop_smooth02/diag_scale_${lam}_result.txt" \
    0 0 0.2 \
    rl_runs/open_laptop_smooth02/diag_videos "scale_${lam}" 0 \
    --rl_enabled true \
    --rl_mode eval_residual_scale_sweep \
    --rl_load_rl_checkpoint_path "$CKPT" \
    --rl_replay_path rl_runs/open_laptop_smooth02/base_replay.pt \
    --rl_residual_scale "$lam" \
    --rl_deterministic_eval true \
    --rl_action_diff_log_path "rl_runs/open_laptop_smooth02/diag_scale_${lam}_action_diff.csv"
done
```

### 8.5 beta=10 checkpoint sweep

```bash
CKPT=rl_runs/open_laptop_smooth02/rl_checkpoint_1200_beta10_alpha005_lr5e5.pt
for lam in 0.0 0.1 0.25 0.5 1.0; do
  bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
    Humanoid-Open-Laptop-v0 2 1 0.2 1 1 \
    "rl_runs/open_laptop_smooth02/diag_beta10_scale_${lam}_result.txt" \
    0 0 0.2 \
    rl_runs/open_laptop_smooth02/diag_videos "beta10_scale_${lam}" 0 \
    --rl_enabled true \
    --rl_mode eval_residual_scale_sweep \
    --rl_load_rl_checkpoint_path "$CKPT" \
    --rl_replay_path rl_runs/open_laptop_smooth02/base_replay.pt \
    --rl_residual_scale "$lam" \
    --rl_deterministic_eval true \
    --rl_action_diff_log_path "rl_runs/open_laptop_smooth02/diag_beta10_scale_${lam}_action_diff.csv"
done
```

### 8.6 关键时间窗口统计

如果要聚焦视频中 7-10 秒附近，先用视频帧/rollout step 对齐确认窗口，再用：

```text
--rl_action_diff_step_start <start_step>
--rl_action_diff_step_end <end_step>
```

注意：当前视频保存 fps 与仿真 step 不是严格同一个概念，不要未经校准就把视频秒数直接当 rollout step。

## 9. 短期实验顺序

1. 当前环境 baseline 复验。
   - baseline 不成功：停止，先查环境/asset/randomize。

2. identity actor。
   - baseline 成功但 identity 失败：停止，先修 RL 插入、normalizer、postprocess、pack/unpack。
   - identity 误差非零：先解释误差来源，不要继续训练。

3. tiny-noise sensitivity。
   - 0.01/0.03 noise 就失败：Open-Laptop 极敏感，先降低 actor freedom 或换任务验证。

4. residual scale sweep。
   - lambda=0 必须约等于 identity。
   - 观察 0.1/0.25/0.5/1.0 的 success 与 per-group residual。

5. per-group residual 分析。
   - 重点检查接触关键阶段 `left/right_ee_pos`、`left/right_ee_rot`。

6. 扩展 paired eval 到 N=10/N=20。
   - 单点结论只用于定位，不用于判断算法优劣。

## 10. 停止继续训练的条件

出现以下任一情况，应停止 RL 训练分析，先修代码/评测：

- 当前 baseline 同一点不能复现成功。
- identity actor 与 baseline 行为显著不一致。
- identity `identity_max_abs_error` 大到足以改变控制。
- `eval_rl`/scale mode 没有加载指定 checkpoint。
- action diff 日志显示 pack/unpack group 顺序错位。
- tiny-noise 在极小尺度下大面积破坏成功，但后续还在全维 total-action 上训练。

## 11. 可以继续做 RL 方案改进的条件

满足以下条件后，才有资格讨论改 loss/actor/reward：

- baseline 当前环境可复现。
- identity actor paired eval 与 baseline 一致。
- tiny-noise 没有在极小尺度下立即摧毁任务，或已接受“任务极敏感”的约束。
- residual scale sweep 显示非零 lambda 的行为模式可解释。
- per-group residual 定位到具体维度/阶段。
- paired eval 统计支持单点视频观察。

在这些条件满足前，不建议继续增加训练步数、只调 beta_ref，或把失败归因为“actor 学坏了”。
