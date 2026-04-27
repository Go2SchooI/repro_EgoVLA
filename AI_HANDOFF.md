# AI Handoff

This file is the fastest way to bring a new AI collaborator up to speed on the current state of this repo.

If a new conversation starts, the AI should read this file first, then read [TRAINING_HANDOFF.md](/root/gpufree-data/EgoVLA_Release/TRAINING_HANDOFF.md) only if training-specific detail is needed.

## Goal

The current priority is:

- Keep the training workflow usable in `vila`
- Keep IsaacLab evaluation usable in `egovla-sim`
- Avoid unnecessary edits to the VILA environment or training stack

## Repo And Environments

- Repo root: `/root/gpufree-data/EgoVLA_Release`
- Training env: `conda activate vila`
- Simulation eval env: `conda activate egovla-sim`

Rule:

- Use `vila` for training and data prep
- Use `egovla-sim` for benchmark evaluation only

## Read First

- [cmd.md](/root/gpufree-data/EgoVLA_Release/cmd.md): short command notes
- [TRAINING_HANDOFF.md](/root/gpufree-data/EgoVLA_Release/TRAINING_HANDOFF.md): detailed training handoff
- [human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh): main eval shell entry
- [human_plan/ego_bench_eval/ik_agent_30hz.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/ik_agent_30hz.py): main eval Python entry
- [human_plan/ego_bench_eval/smoke_test_env.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/smoke_test_env.py): fast environment smoke test

## Current Working Eval Status

Formal benchmark evaluation is working end-to-end in `egovla-sim`.

Verified successful run:

```bash
cd /root/gpufree-data/EgoVLA_Release
conda activate egovla-sim
EGO_VLA_EVAL_DEVICE=cuda:0 EGO_VLA_SIGLIP_ATTN_IMPLEMENTATION=sdpa \
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
Humanoid-Push-Box-v0 1 2 0.2 3 1 result_log.txt 0 0 0.8 video_output evaluation_tag
```

Successful signals:

- `env_created`
- `initial_env_reset_done`
- `warmup_done`
- `rollout_start`
- progress bars reaching `400/400`
- final `Simulation App Shutting Down`
- no Python traceback

Outputs:

- result log: [result_log.txt](/root/gpufree-data/EgoVLA_Release/result_log.txt)
- videos: [video_output](/root/gpufree-data/EgoVLA_Release/video_output)

## Current Default Checkpoint

The current eval default checkpoint is:

- `/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v5-from14000/checkpoint-3500`

This was set intentionally for evaluation.

## Important Eval Behavior

### Device Rule

Inside the current container/session, only one visible CUDA device is often exposed to the process.

That means:

- Use `EGO_VLA_EVAL_DEVICE=cuda:0`
- Do not assume `cuda:1` is valid even if the host machine has multiple GPUs

The eval code now canonicalizes invalid CUDA ordinals and falls back to `cuda:0` when needed.

### Attention Backend Rule

The environment does not require `flash-attn`.

Current safe choice:

- `EGO_VLA_SIGLIP_ATTN_IMPLEMENTATION=sdpa`

Impact:

- This mainly affects speed and possibly memory use
- It should not break the correctness of evaluation

### Long Startup Rule

IsaacLab evaluation can appear stuck during:

- scene creation
- simulation startup
- first camera/render warmup

This is why progress prints were added to `ik_agent_30hz.py`.

If the log is moving through:

- `creating_env`
- `env_created`
- `initial_env_reset_done`
- `warmup_progress`

then it is not hard-dead; it is still progressing.

## Important Eval Parameters

The shell interface is:

```bash
bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
TASK ROOM_IDX TABLE_IDX SMOOTH_WEIGHT NUM_EPISODES NUM_TRIALS \
RESULT_PATH SAVE_FRAMES PROJECT_TRAJS HAND_SMOOTH_WEIGHT VIDEO_PATH ADDITIONAL_LABEL
```

Most commonly changed fields:

- `TASK`: task name such as `Humanoid-Push-Box-v0`
- `ROOM_IDX`: room/background choice
- `TABLE_IDX`: table/layout choice
- `SMOOTH_WEIGHT`: end-effector smoothing during eval
- `NUM_EPISODES`: number of benchmark episodes to run
- `NUM_TRIALS`: number of trials per episode
- `HAND_SMOOTH_WEIGHT`: hand-action smoothing during eval

Notes:

- `smooth_weight` and `hand_smooth_weight` are eval-only postprocessing parameters, not training parameters
- `num_episodes` selects the first N predefined benchmark episodes for that task
- `num_trials` reruns each episode under different randomization

## Video Output

Generated evaluation videos are now automatically transcoded to H.264 after each episode video finishes writing.

Implementation:

- [ik_agent_30hz.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/ik_agent_30hz.py)

Behavior:

- OpenCV still writes an intermediate MP4
- The script then calls `ffmpeg`
- The final file keeps the same `.mp4` name
- Logs include:
  - `h264_transcode_start`
  - `h264_transcode_done`

This requires `ffmpeg` to exist in the environment. It is already available on the current machine.

## Current Compatibility Fixes Already Applied

The following issues were already resolved and should not be re-opened unless regression appears:

- headless Vulkan / EGL setup for IsaacSim
- boto stack preload from `egovla-sim`
- CUDA device canonicalization for eval
- fast smoke-test mode in `smoke_test_env.py`
- eval progress logging in `ik_agent_30hz.py`
- eval model loading without pulling training-only code paths
- delayed or guarded imports for training-only VILA modules
- `datasets` import removed from eval path in retarget modules
- MANO path and NumPy compatibility fixes
- SigLIP attention fallback to `sdpa` when `flash-attn` is unavailable
- `VisionTower` no longer incorrectly imports `s2wrapper` in non-S2 mode
- auto H.264 video transcode after eval rollout

## Files That Were Touched For Eval Stability

High-signal files:

- [human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh)
- [human_plan/ego_bench_eval/ik_agent_30hz.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/ik_agent_30hz.py)
- [human_plan/ego_bench_eval/smoke_test_env.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/smoke_test_env.py)
- [human_plan/ego_bench_eval/utils.py](/root/gpufree-data/EgoVLA_Release/human_plan/ego_bench_eval/utils.py)
- [human_plan/vila_eval/utils/load_model.py](/root/gpufree-data/EgoVLA_Release/human_plan/vila_eval/utils/load_model.py)
- [human_plan/vila_eval/utils/eval_func.py](/root/gpufree-data/EgoVLA_Release/human_plan/vila_eval/utils/eval_func.py)
- [human_plan/utils/mano/model.py](/root/gpufree-data/EgoVLA_Release/human_plan/utils/mano/model.py)
- [human_plan/utils/nn_retarget.py](/root/gpufree-data/EgoVLA_Release/human_plan/utils/nn_retarget.py)
- [human_plan/utils/nn_retarget_tomano.py](/root/gpufree-data/EgoVLA_Release/human_plan/utils/nn_retarget_tomano.py)
- [human_plan/utils/nn_retarget_formano.py](/root/gpufree-data/EgoVLA_Release/human_plan/utils/nn_retarget_formano.py)
- [VILA/llava/model/language_model/llava_llama.py](/root/gpufree-data/EgoVLA_Release/VILA/llava/model/language_model/llava_llama.py)
- [VILA/llava/model/llava_arch.py](/root/gpufree-data/EgoVLA_Release/VILA/llava/model/llava_arch.py)
- [VILA/llava/model/multimodal_encoder/siglip_encoder.py](/root/gpufree-data/EgoVLA_Release/VILA/llava/model/multimodal_encoder/siglip_encoder.py)
- [VILA/llava/model/multimodal_encoder/vision_encoder.py](/root/gpufree-data/EgoVLA_Release/VILA/llava/model/multimodal_encoder/vision_encoder.py)
- [VILA/llava/data/__init__.py](/root/gpufree-data/EgoVLA_Release/VILA/llava/data/__init__.py)

## Common Pitfalls

### Old Isaac/Eval Processes Still Running

Before rerunning eval, check for leftovers:

```bash
pgrep -af 'ik_agent_30hz.py|smoke_test_env.py|simulation_app.py|isaaclab.python.headless'
```

If needed:

```bash
pkill -9 -f 'human_plan/ego_bench_eval/ik_agent_30hz.py'
pkill -9 -f 'human_plan/ego_bench_eval/smoke_test_env.py'
pkill -9 -f 'omni/isaac/kit/simulation_app.py'
```

Then verify again with `pgrep`.

### Misreading Warnings As Hard Failures

The following warnings have appeared during successful runs and are not necessarily blockers:

- GLFW initialization failed in headless mode
- `pxr.Semantics is deprecated`
- `Not all actuators are configured! 38 != 50`
- Gymnasium wrapper deprecation warnings
- MANO shape-coefficient warnings

The real signal is whether rollout progresses and whether the process ends with a traceback.

## Smoke Test

The environment smoke test is:

```bash
cd /root/gpufree-data/EgoVLA_Release
conda activate egovla-sim
python human_plan/ego_bench_eval/smoke_test_env.py --task Humanoid-Push-Box-v0 --headless --device cuda:0
```

Success marker:

- `smoke_test_ok True`

Default smoke behavior now uses a fast mode to avoid very slow full-scene initialization.

## Training Note

Training details are intentionally not duplicated here.

If the new AI needs to touch training:

- Read [TRAINING_HANDOFF.md](/root/gpufree-data/EgoVLA_Release/TRAINING_HANDOFF.md)
- Respect the distinction between true resume and checkpoint initialization

## Recommended New-Chat Prompt

If the user opens a new conversation, a good first instruction is:

```text
Please read /root/gpufree-data/EgoVLA_Release/AI_HANDOFF.md first, then help me continue from the current EgoVLA_Release state without undoing existing eval compatibility fixes.
```
