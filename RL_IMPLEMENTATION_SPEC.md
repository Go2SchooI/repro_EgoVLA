# EgoVLA Simulation RL Post-Training Implementation Spec

Status: technical spec only. Do not implement the full RL stack in this pass.

Primary constraint: when `rl.enabled=false`, the current EgoVLA evaluation path must be behaviorally unchanged.

## 1. Current Action Path Trace

Relevant files:

- `human_plan/ego_bench_eval/ik_agent_30hz.py`
- `human_plan/ego_bench_eval/utils.py`
- `human_plan/vila_eval/utils/eval_func.py`
- `human_plan/vila_eval/utils/load_model.py`
- `VILA/llava/model/language_model/llava_llama.py`
- `VILA/llava/model/ego_vla_decoder/traj_decoder.py`
- `VILA/llava/model/ego_vla_decoder/transformer.py`
- IsaacLab benchmark env: `/root/gpufree-data/IsaacLab/Ego_Humanoid_Manipulation_Benchmark/source/extensions/humanoid.tasks/humanoid/tasks/base_env/base_env.py`

### 1.1 Raw EgoVLA Prediction

The eval shell `human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh` currently sets:

- `--predict_future_step 30`
- `--future_index 1`
- `--traj_action_output_dim 48`
- `--traj_decoder_type transformer_split_action_v2`
- `--raw_action_label True`
- `--sep_proprio True`
- `--sep_query_token True`
- `--traj_action_output_ee_dim 6`
- `--traj_action_output_hand_dim 30`
- `--traj_action_output_ee_rot_dim 12`
- `--traj_action_output_ee_2d_dim 0`

`human_plan/vila_eval/utils/eval_func.py::eval_single_sample` calls `model.forward(**data_dict)` with `inference=True` and returns:

```text
output.prediction.cpu().numpy()
```

Current expected prediction shape:

```text
pred: (T, 48), T = data_args.predict_future_step = 30
```

The 48D prediction is consumed by current eval code as:

```text
pred[:, 0:6]    -> pred_3d, shape (T, 2, 3)
pred[:, 6:36]   -> pred_hand, shape (T, 2, 15)
pred[:, 36:48]  -> pred_rot6d, shape (T, 2, 6), then rot6d_to_rotmat -> (T, 2, 3, 3)
```

Important: do not infer the runtime action layout from the names `output_projection_left/right` in `transformer.py`. The existing sim eval path treats model output as `ee_3d(6) + mano_hand(30) + ee_rot6d(12)`, and this is the authoritative interface for the current checkpoint/eval stack.

### 1.2 `ik_eval_single_step` Return Values

`human_plan/ego_bench_eval/utils.py::ik_eval_single_step` converts raw 48D model output into the command components later used by IK:

```text
action_dict["left_ee_pose"]          shape (T, 7)
action_dict["right_ee_pose"]         shape (T, 7)
action_dict["left_qpos_multi_step"]  shape (T, 12)
action_dict["right_qpos_multi_step"] shape (T, 12)
```

Semantics:

- `left_ee_pose`, `right_ee_pose`: IsaacLab/world-frame end-effector target poses, `[x, y, z, qw, qx, qy, qz]`.
- `left_qpos_multi_step`, `right_qpos_multi_step`: 12D Inspire robot hand joint targets from MANO prediction through `hand_actuation_net`.

Other returned fields:

- `left_qpos`, `right_qpos`: first step only, shape `(12,)`; not used in the rollout after smoothing.
- `left_ee_trans_cam`, `right_ee_trans_cam`: model-predicted camera-frame MANO translations, shape `(T, 3)`.
- `pred_3d`: camera-frame predicted translations, shape `(T, 2, 3)`, used for optional trajectory visualization.

### 1.3 Repeat And Temporal Smoothing

In `human_plan/ego_bench_eval/ik_agent_30hz.py`, after `ik_eval_single_step`:

```python
action_hist_right_ee.append(repeat_action(action_dict["right_ee_pose"], data_args.future_index))
action_hist_left_ee.append(repeat_action(action_dict["left_ee_pose"], data_args.future_index))
action_hist_left_hand.append(repeat_action(action_dict["left_qpos_multi_step"], data_args.future_index))
action_hist_right_hand.append(repeat_action(action_dict["right_qpos_multi_step"], data_args.future_index))
```

With current `future_index=1`, `repeat_action` does not change the length:

```text
right_ee repeated:  (30, 7)
left_ee repeated:   (30, 7)
left_hand repeated: (30, 12)
right_hand repeated:(30, 12)
```

In general:

```text
repeat_action((T, D), repeat=future_index) -> (T * future_index, D)
hist_len = predict_future_step * future_index
```

`smooth_action(hist_len, smooth_weight, action_deque)` selects one aligned future step from each historical chunk and returns a single command:

```text
action_left_ee:    (7,)
action_right_ee:   (7,)
action_left_hand:  (12,)
action_right_hand: (12,)
```

### 1.4 `ik_step` Input And Final `env.step(action)`

`human_plan/ego_bench_eval/utils.py::ik_step` receives the smoothed command parts:

```text
left_ee_goal:   (7,)
right_ee_goal:  (7,)
left_hand_dof:  (12,)
right_hand_dof: (12,)
```

`ik_step` then:

- sets left/right world-frame EE pose goals,
- converts them to robot frame,
- runs Differential IK for the 7D left arm and 7D right arm joint targets,
- writes 12D left and right hand targets into the robot action tensor.

The benchmark `BaseEnvCfg` defines:

```text
num_actions = 50
```

So `env.step(action)` receives:

```text
action: torch.Tensor, shape (num_envs, 50), currently num_envs=1
```

This 50D tensor is not the RL actor output. It is the env joint-target tensor filled by IK/hand retargeting. The actor insertion point is before `ik_step`, so actor action dim is 38, not 48 and not 50.

## 2. Actor Insertion Point

Insert the RL actor after temporal smoothing and before `ik_step(...)`:

```text
EgoVLA raw chunk
  -> ik_eval_single_step(...)
  -> repeat_action(...)
  -> smooth_action(...)
  -> pack_action(...) gives a_ref
  -> Gaussian actor gives a_exec
  -> postprocess/unpack_action(...)
  -> ik_step(...)
  -> env.step(action)
```

`a_ref`:

- The command that the original pipeline would have passed to `ik_step` at the current simulation step.
- It is computed after temporal smoothing and before any RL actor intervention.

`a_exec`:

- The actor-produced total single-step command in the same packed 38D command space as `a_ref`.
- It is not a residual.
- It is not a full action chunk.
- After postprocessing, it is unpacked and passed to `ik_step`.

## 3. Action Packing Interface

Add small helper functions, preferably in a new RL utility module or `human_plan/ego_bench_eval/utils.py` if keeping the first patch minimal:

```python
ACTION_SLICES = {
    "left_ee": slice(0, 7),
    "right_ee": slice(7, 14),
    "left_hand": slice(14, 26),
    "right_hand": slice(26, 38),
}

def pack_action(left_ee, right_ee, left_hand, right_hand) -> np.ndarray:
    ...

def unpack_action(action: np.ndarray) -> dict[str, np.ndarray]:
    ...
```

Packed layout:

```text
left_ee:    [0:7]    (x, y, z, qw, qx, qy, qz)
right_ee:   [7:14]   (x, y, z, qw, qx, qy, qz)
left_hand:  [14:26]  12 Inspire hand DOF targets
right_hand: [26:38]  12 Inspire hand DOF targets
```

Derived value:

```text
actor_action_dim = dim(a_ref) = 38
```

Do not hard-code `48` for the actor. Use `len(pack_action(...))` or a single `ACTION_DIM = 38` assert derived from slices.

### 3.1 Action Normalization

Reference regularization should be computed in normalized packed-command space:

```text
||a_exec_norm - a_ref_norm||^2
```

First-version recommendation:

- Store and train actor distributions in normalized 38D command space.
- Denormalize the sampled total command before `unpack_action` and `ik_step`.
- Compute mean/std from `a_ref` collected in `rl.mode=collect_base`, then reuse the saved stats for `offline_rl`, `online_rl`, and `eval_rl`.
- Keep stats per packed dimension, not per semantic group only.

This still satisfies "actor outputs total action": the sampled normalized vector represents the total command, not a residual; denormalization maps it to the actual IK command space.

### 3.2 Postprocess Before `ik_step`

After actor sampling and denormalization:

- Normalize both EE quaternions:
  - left quaternion slice inside packed action: `left_ee[3:7]`
  - right quaternion slice inside packed action: `right_ee[3:7]`
- If `dot(q_exec, q_ref) < 0`, flip quaternion sign to keep continuity.
- Clamp non-finite values back to `a_ref`.
- Clamp hand DOFs to env robot joint limits if available:
  - `env.robot_dof_lower_limits[env.cfg.left_hand_cfg.joint_ids]`
  - `env.robot_dof_upper_limits[env.cfg.left_hand_cfg.joint_ids]`
  - same for right hand.
- Add optional action clipping around normalized range, e.g. tanh-squashed Gaussian or configurable `rl.action_norm_clip`.

## 4. `h_t` Feature Hook Design

Preferred hook:

```text
h_t = traj decoder input latent
```

Concrete location:

- `VILA/llava/model/language_model/llava_llama.py`
- right after:

```python
action_output = outputs.hidden_states[-1][output_mask]
```

Current shape before decoder:

```text
flat action_output: (B * Q, H)
```

Current eval values:

```text
B = 1
Q = 2 * predict_future_step = 60
H = hidden_size = 1536
```

Why Q is 60:

- The current placeholder/action query path creates two action query tokens per future step under the current `raw_action_label=True`, `input_placeholder_diff_index=True`, `sep_query_token=True` setup.
- The decoder reshapes this flat latent using `raw_proprio_inputs.shape[0]` as batch size.

Recommended returned feature shape:

```text
h_in: (B, Q, H)
```

Secondary hook for ablation:

```text
h_preout = action expert output-projection input latent
```

Concrete location:

- `VILA/llava/model/ego_vla_decoder/transformer.py`
- after the transformer layers and after stripping proprio tokens:

```python
if self.use_proprio:
    if self.sep_proprio:
        latent = latent[:, 6:, :]
    else:
        latent = latent[:, 1:, :]
```

Shape:

```text
h_preout: (B, Q, H)
```

### 4.1 Return API

Add a config-gated inference API:

```python
eval_single_sample(..., return_rl_features=False)
ik_eval_single_step(..., return_rl_features=False)
model.forward(..., return_rl_features=False)
TrajDecoder.forward/inference(..., return_rl_features=False)
TransformerSplitActV2.forward/inference(..., return_rl_features=False)
```

When disabled, output and behavior must be exactly as today.

When enabled, return an additional dict:

```python
{
    "prediction": pred,
    "rl_features": {
        "h_in": h_in.detach(),
        "h_preout": h_preout.detach(),  # optional based on rl.feature_hook
    },
}
```

Implementation detail:

- `h_in` can be reshaped in `llava_llama.py` before calling `traj_decoder`, because `raw_proprio_inputs.shape[0]` gives `B`.
- `h_preout` is easiest to expose from `TransformerSplitActV2`.
- Always `.detach()` before returning features.
- Keep collection under `torch.inference_mode()` in eval/collector.
- Also call `model.eval()` and set `requires_grad_(False)` for all EgoVLA parameters when `rl.freeze_egovla=true`.

### 4.2 Actor Feature Pooling

Store raw `h_in` as `(Q, H)` in replay raw fields for reproducibility.

For first-version actor input, use:

```text
h_feat = mean_pool(h_in, dim=query_token) -> (H,)
```

With the current checkpoint:

```text
h_feat_dim = 1536
```

Do not flatten `(60, 1536)` into the default actor input in v1; keep that as an ablation option.

## 5. Base Chunk Summary

Base chunk should be summarized in the same 38D command space as the actor output.

Build an unsmoothed packed base chunk from `action_dict`:

```python
base_chunk[t] = pack_action(
    action_dict["left_ee_pose"][t],
    action_dict["right_ee_pose"][t],
    action_dict["left_qpos_multi_step"][t],
    action_dict["right_qpos_multi_step"][t],
)
```

Current shape:

```text
base_chunk: (30, 38)
```

First-version summary:

```text
selected steps = [0, 5, 10, 20, 29]
base_chunk_summary = base_chunk[selected_steps].reshape(-1)
```

For shorter chunks, use `min(step_idx, T - 1)`.

Current summary dim:

```text
5 * 38 = 190
```

Keep optional alternatives behind config:

- `selected_steps`
- `first_last_delta_mean`: `[first, last, last-first, mean]`, dim `4 * 38 = 152`
- `flatten_full_chunk`, dim `T * 38 = 1140` for current T=30

## 6. Actor/Critic Observations

### 6.1 Actor Obs

First-version actor obs:

```text
actor_obs = concat([
    h_feat,
    proprio,
    base_chunk_summary,
    a_ref_norm or a_ref
])
```

Recommended use:

- Use `a_ref_norm` in actor obs if the actor outputs normalized actions.
- Use `proprio = raw_data_dict["proprio_input"]`.

Current expected dimensions:

```text
h_feat:             1536
proprio:            16 if data_args.input_hand_dof=False
                    42 if data_args.input_hand_dof=True
base_chunk_summary: 190
a_ref_norm:         38
actor_obs_dim:      1780 with current default proprio=16
```

The exact actor obs dim should be derived at runtime and asserted.

### 6.2 Critic Obs

First-version critic uses privileged simulator state:

```text
critic_obs = x_t
```

Recommended helper:

```python
build_priv_state(env, obs_dict) -> np.ndarray
```

Include:

- `qpos`
- `qvel`
- previous `action`
- `left_ee_pose`, `right_ee_pose`
- `left_target_ee_pose`, `right_target_ee_pose`
- `left_finger_tip_pos`, `right_finger_tip_pos`
- contact force fields
- task-specific non-image object states if present in `obs_dict`

Exclude:

- image/depth tensors: keys containing `rgb`, `fixed_d`, `distance`, camera image data
- success/progress labels from critic input in v1: keys containing `success`
- reward/done fields

Flatten all included tensors for env 0 and concatenate. Save the included key list with the replay/normalization metadata so offline replay shape is reproducible.

## 7. Replay Buffer Design

Use two layers.

### 7.1 Raw Reproducible Fields

Store these per transition:

```text
h_in
h_preout                  optional
proprio
base_chunk
a_ref
priv_state
a_exec
reward
done
success
timeout
next_h_in
next_h_preout             optional
next_proprio
next_base_chunk
next_a_ref
next_priv_state
```

Notes:

- `h_in`, `h_preout`, `base_chunk`, and `a_ref` must be detached from EgoVLA.
- Raw fields are for ablation/debug/rebuilding fast fields.
- Raw fields should be saved in a format that preserves shapes, e.g. `.npz`, HDF5, or a typed torch checkpoint with metadata.

### 7.2 Fast Training Fields

Replay updates should read only:

```text
actor_obs
critic_obs
action            # a_exec, normalized if actor/critic train in normalized action space
ref_action        # a_ref, same normalized space as action for ref regularization
reward
next_actor_obs
next_critic_obs
done
```

Principles:

- Default replay update must not rerun EgoVLA.
- Fast fields can be precomputed during collection or lazily built once when loading replay.
- Save action normalizer stats, actor obs normalizer stats, critic obs normalizer stats, chunk summary config, feature hook config, and privileged-state key list next to the replay.

## 8. SAC-Style Loss

Actor is a Gaussian policy over normalized packed action space:

```text
pi(a_norm | actor_obs)
```

Denormalize `a_norm` to `a_exec` only for environment execution.

Critics:

```text
Q1(critic_obs, a_norm)
Q2(critic_obs, a_norm)
```

Target:

```text
a_next_norm, logp_next = actor(next_actor_obs)
target_q = reward + gamma * (1 - done) * (
    min(Q1_target(next_critic_obs, a_next_norm),
        Q2_target(next_critic_obs, a_next_norm))
    - alpha_entropy * logp_next
)
```

Critic loss:

```text
L_Q = MSE(Q1(critic_obs, action_norm), target_q)
    + MSE(Q2(critic_obs, action_norm), target_q)
```

Actor loss:

```text
L_actor = mean(
    alpha_entropy * log_prob
    - min(Q1(critic_obs, a_pi_norm), Q2(critic_obs, a_pi_norm))
    + beta_ref * ||a_pi_norm - a_ref_norm||^2
)
```

First version:

- Fixed `alpha_entropy` from config.
- Fixed `beta_ref` from config.
- No automatic entropy tuning unless added later behind config.
- `reference-action dropout` interface exists but default `ref_dropout_p=0.0`.

## 9. Reward And Done

Use sparse terminal success:

```text
success = env_results[0]["success"].sum().item() == 1
timeout = (step_idx + 1) >= max_horizon
reward = 1.0 if success else 0.0
done = success or timeout
```

This `done` is the RL episode done, independent of IsaacLab `BaseEnv._get_dones`, which currently only returns timeout.

On success, current eval breaks immediately. RL collection should store the successful terminal transition first, then end/reset.

## 10. Config And CLI

Canonical nested config keys:

```text
rl.enabled = false
rl.mode = collect_base | offline_rl | online_rl | eval_rl | debug_trace_action_path
rl.actor_insert_point = after_temporal_smoothing
rl.feature_hook = traj_decoder_input | pre_output
rl.freeze_egovla = true
rl.cache_rl_features = true
rl.reward_type = sparse_success
rl.beta_ref
rl.alpha_entropy
rl.gamma
rl.tau
rl.batch_size
rl.replay_capacity
rl.ref_dropout_p = 0.0
rl.deterministic_eval = true
rl.debug_dump_shapes = false
rl.save_debug_transition_path
rl.action_normalizer_path
rl.replay_path
rl.actor_checkpoint_path
rl.critic_checkpoint_path
rl.chunk_summary_type = selected_steps
rl.chunk_summary_steps = [0, 5, 10, 20, 29]
```

Because the current entry point uses `HfArgumentParser` plus argparse additions, implementation can expose flat CLI names while mapping them to nested config internally:

```text
--rl_enabled
--rl_mode
--rl_actor_insert_point
--rl_feature_hook
--rl_freeze_egovla
--rl_cache_rl_features
--rl_reward_type
--rl_beta_ref
--rl_alpha_entropy
--rl_gamma
--rl_tau
--rl_batch_size
--rl_replay_capacity
--rl_ref_dropout_p
--rl_deterministic_eval
--rl_debug_dump_shapes
--rl_save_debug_transition_path
--rl_action_normalizer_path
--rl_replay_path
--rl_actor_checkpoint_path
--rl_critic_checkpoint_path
```

Default behavior must match current eval:

```text
rl.enabled=false
```

## 11. Files To Modify In Implementation Pass

Minimal code path changes:

- `human_plan/ego_bench_eval/ik_agent_30hz.py`
  - parse RL args
  - freeze EgoVLA when enabled
  - call `ik_eval_single_step(..., return_rl_features=...)`
  - build `base_chunk`, `a_ref`, actor obs, critic obs
  - route `a_exec` to `ik_step` only when RL enabled
  - collect replay transitions
  - preserve exact existing branch when disabled

- `human_plan/ego_bench_eval/utils.py`
  - add `pack_action`, `unpack_action`, `postprocess_packed_action`
  - add optional action-path debug shape helper
  - extend `ik_eval_single_step` to return raw `pred` and `rl_features` when requested

- `human_plan/vila_eval/utils/eval_func.py`
  - add `return_rl_features=False`
  - return prediction plus feature dict only when requested

- `VILA/llava/model/language_model/llava_llama.py`
  - add `return_rl_features=False`
  - expose detached `h_in`
  - pass feature flag to traj decoder

- `VILA/llava/model/ego_vla_decoder/traj_decoder.py`
  - pass through `return_rl_features`

- `VILA/llava/model/ego_vla_decoder/transformer.py`
  - expose detached `h_preout` when requested

New RL modules:

- `human_plan/ego_bench_eval/rl/config.py`
- `human_plan/ego_bench_eval/rl/action_space.py`
- `human_plan/ego_bench_eval/rl/features.py`
- `human_plan/ego_bench_eval/rl/replay_buffer.py`
- `human_plan/ego_bench_eval/rl/actor_critic.py`
- `human_plan/ego_bench_eval/rl/sac.py`
- `human_plan/ego_bench_eval/rl/privileged_state.py`

Optional shell:

- `human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh`
  - add env-pass-through RL args only if needed; default should remain identical.

## 12. Runtime Asserts And Risk Points

Required asserts:

- `pred.ndim == 2`
- `pred.shape[-1] == 48` for current EgoVLA output conversion in `ik_eval_single_step`
- `action_dict["left_ee_pose"].shape[-1] == 7`
- `action_dict["right_ee_pose"].shape[-1] == 7`
- `action_dict["left_qpos_multi_step"].shape[-1] == 12`
- `action_dict["right_qpos_multi_step"].shape[-1] == 12`
- after repeat: first dim equals `predict_future_step * future_index`
- after smoothing:
  - `action_left_ee.shape == (7,)`
  - `action_right_ee.shape == (7,)`
  - `action_left_hand.shape == (12,)`
  - `action_right_hand.shape == (12,)`
- `pack_action(...).shape == (38,)`
- `actor_action_dim == 38`
- actor output shape equals `(38,)`
- no NaN/Inf in `a_ref`, `a_exec`, `actor_obs`, `critic_obs`
- quaternion norms after postprocess are near 1
- `action.shape == (env.scene.num_envs, env.num_actions)`
- `env.num_actions == 50` for this benchmark version
- `rl.actor_insert_point == "after_temporal_smoothing"` in v1
- if `rl.enabled=false`, no actor/replay/model feature-hook side effects are executed

Risk points:

- Actor dim is 38 at the insertion point; using 48 would train in the wrong space.
- Actor should not output final 50D env joint target; IK remains part of the control pipeline.
- Current 48D model prediction layout is eval-code-defined as `ee_3d + mano_hand + rot6d`; do not reorder it because of decoder variable names.
- Direct Gaussian quaternion output can create invalid rotations; normalize and sign-align quaternions before `ik_step`.
- Replay update must not rerun EgoVLA; cache fast fields.
- Feature tensors must be detached; EgoVLA must receive no RL gradients.
- `proprio_input` dim depends on `data_args.input_hand_dof`; derive at runtime.
- Privileged critic obs must have a saved key list and fixed flatten order; otherwise offline replay becomes non-reproducible.
- Success labels should be used for reward/done, not included in critic obs by default.
- Keep all new behavior config-gated and disabled by default.

## 13. Debug Trace Mode

Implement `rl.mode=debug_trace_action_path` as a non-training mode that runs one or a few rollout steps and dumps shapes/metadata to `rl.save_debug_transition_path`.

Suggested dump:

```text
pred_shape
left_ee_pose_shape
right_ee_pose_shape
left_qpos_multi_step_shape
right_qpos_multi_step_shape
repeated_shapes
smoothed_shapes
packed_a_ref_shape
base_chunk_shape
base_chunk_summary_shape
h_in_shape
h_preout_shape if available
proprio_shape
actor_obs_shape
critic_obs_shape
env_num_actions
env_action_shape
success
timeout
```

This mode should call the same helpers as RL collection but can skip actor loading and use `a_exec=a_ref`.
