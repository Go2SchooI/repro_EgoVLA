# Training Handoff

This file is for quickly bringing a new AI collaborator up to speed on the training workflow in this repo.

## Goal

The current project use case is robot-data fine-tuning on processed EgoVLA humanoid simulation data.

The most important rule is:

- Same `RUN_NAME/output_dir` means true resume.
- New `RUN_NAME` plus `MODEL_PATH=/path/to/checkpoint-*` means checkpoint initialization for a new run.

These are not the same thing.

## Repo And Environments

- Repo root: `/root/gpufree-data/EgoVLA_Release`
- Main training env: `conda activate vila`
- Simulation eval env: `conda activate egovla-sim`

Use `vila` for training and dataset preprocessing.
Use `egovla-sim` only for IsaacLab simulation benchmark evaluation.

## Training Entrypoints

Main training shell:

- [training_scripts/robot_finetuning/subset_train_wandb.sh](/root/gpufree-data/EgoVLA_Release/training_scripts/robot_finetuning/subset_train_wandb.sh)

Smoke-test training shell:

- [training_scripts/robot_finetuning/smoke_train.sh](/root/gpufree-data/EgoVLA_Release/training_scripts/robot_finetuning/smoke_train.sh)

Python trainer:

- [human_plan/vila_train/train_mem.py](/root/gpufree-data/EgoVLA_Release/human_plan/vila_train/train_mem.py)
- [human_plan/vila_train/train.py](/root/gpufree-data/EgoVLA_Release/human_plan/vila_train/train.py)

DeepSpeed config used by training:

- [VILA/scripts/zero3.json](/root/gpufree-data/EgoVLA_Release/VILA/scripts/zero3.json)

## How Resume Logic Actually Works

`train.py` checks `training_args.output_dir` for `checkpoint-*`.

Behavior:

- If `output_dir` already contains checkpoints, training tries to resume from the latest checkpoint under that directory.
- If `output_dir` is new, `MODEL_PATH` is used only as initialization.

Meaning:

- `MODEL_PATH=/some/checkpoint-2000` plus a new `RUN_NAME` loads model weights only.
- Optimizer state, scheduler state, and global step do not carry over in that case.
- Exact resume only happens when the run continues inside the same checkpoint directory tree.

## Current Data Layout

Raw robot data:

- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM`

Current raw task folders:

- `Close-Drawer`
- `Flip-Mug`
- `Insert-And-Unload-Cans`
- `Open-Laptop`
- `Pour-Balls`

There is also a `preview_videos` directory under the raw data root.

Processed training data:

- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/HF_images`
- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/HF_hand_FIXED_SET_MIX_train`
- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/hf_images_mapping.pkl`

Intermediate preprocessing outputs:

- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/image_parquets`
- `/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/hand_FIXED_SET_MIX_train_parquets`

`subset_train_wandb.sh` requires exactly these three processed artifacts before training:

- `HF_images`
- `HF_hand_FIXED_SET_MIX_train`
- `hf_images_mapping.pkl`

## Current Training Script Defaults

From [subset_train_wandb.sh](/root/gpufree-data/EgoVLA_Release/training_scripts/robot_finetuning/subset_train_wandb.sh):

- `N_GPU` default: `4`
- `PER_DEVICE_BS` default: `1`
- `GRAD_ACCUM_STEPS` default: `4`
- `NUM_EPOCHS` default: `10`
- `CHECKPOINTS_ROOT` default: `./checkpoints`
- `DATA_ROOT` default: `./data/EgoVLA_SIM_Processed`
- `WANDB_PROJECT_NAME` default: `ego_manip_release`
- learning rate is hard-coded to `2e-5`
- `eval_steps` is hard-coded to `500`
- `save_steps` is hard-coded to `500`
- `save_total_limit` is `3`

Important limitation:

- `learning_rate`, `eval_steps`, and `save_steps` are not currently env-overridable.
- If someone wants to tune them from the shell, the script itself must be edited.

Effective batch size formula:

- `PER_DEVICE_BS * N_GPU * GRAD_ACCUM_STEPS`

Useful examples:

- `1 * 4 * 4 = 16`
- `1 * 6 * 3 = 18`
- `1 * 5 * 4 = 20`

## Current Checkpoint Lineage

Checkpoint root:

- `/root/gpufree-data/EgoVLA_Release/checkpoints`

Relevant directories currently present:

- `mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4`
- `otv-fixed-set-subset-4gpu-wandb`
- `otv-fixed-set-subset-4gpu-wandb-v2-5tasks`
- `otv-fixed-set-subset-6gpu-wandb-v3-from3500`
- `otv-fixed-set-subset-5gpu-wandb-v4-from2000`
- `otv-fixed-set-subset-6gpu-wandb-v5-from14000`
- `otv-fixed-set-smoke-4gpu`
- `otv-fixed-set-smoke-1gpu`

Interpretation:

- `...v2-5tasks` is a later 4-GPU 5-task branch.
- `...v3-from3500` is a new 6-GPU run initialized from a 4-GPU checkpoint.
- `...v4-from2000` is a new 5-GPU run initialized from a 6-GPU checkpoint.
- `...v5-from14000` is a newer 6-GPU run initialized from the 5-GPU branch.

Do not assume lineage by name means exact resume.
Only same `output_dir` resume preserves optimizer and scheduler state.

## Exact Resume Vs New Run

### Exact Resume

Use exact resume when:

- GPU count is unchanged
- You want optimizer state to continue
- You want scheduler state to continue
- You want global step to continue
- You want the same run directory to keep growing

Pattern:

```bash
cd /root/gpufree-data/EgoVLA_Release
conda activate vila

CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPU=4 \
PER_DEVICE_BS=1 \
GRAD_ACCUM_STEPS=4 \
NUM_EPOCHS=6 \
RUN_NAME=otv-fixed-set-subset-4gpu-wandb-v2-5tasks \
WANDB_PROJECT_NAME=ego_manip_release \
bash training_scripts/robot_finetuning/subset_train_wandb.sh
```

If `./checkpoints/$RUN_NAME` already contains checkpoints, this resumes from the latest checkpoint in that directory.

### New Run Initialized From A Checkpoint

Use this when:

- GPU count changes
- You want a clean experiment branch
- You expanded or changed the data and do not want to overwrite the old run

Pattern:

```bash
cd /root/gpufree-data/EgoVLA_Release
conda activate vila

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
N_GPU=5 \
PER_DEVICE_BS=1 \
GRAD_ACCUM_STEPS=4 \
NUM_EPOCHS=6 \
RUN_NAME=otv-fixed-set-subset-5gpu-wandb-v4-from2000 \
MODEL_PATH=/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v3-from3500/checkpoint-2000 \
WANDB_PROJECT_NAME=ego_manip_release \
bash training_scripts/robot_finetuning/subset_train_wandb.sh
```

This loads model weights from `MODEL_PATH`, but resets optimizer state and global step.

## GPU Count Change Rule

This project uses DeepSpeed ZeRO-3.

Practical rule for this repo:

- Same GPU count: exact resume is acceptable.
- Different GPU count: do not rely on exact resume.
- Different GPU count should be treated as checkpoint initialization plus a new run.

If someone says "I want to completely continue the old run" and also wants to change from 4 GPUs to 6 GPUs or 6 GPUs to 5 GPUs, that is a conflict in practice here.

Recommended response:

- If exact continuity matters, keep the old GPU count.
- If the new GPU count matters, create a new run initialized from the desired checkpoint.

## W&B Notes

`subset_train_wandb.sh` expects W&B auth.

It checks:

- `WANDB_API_KEY` in the environment
- or `.wandb_api` in repo root

If neither exists, training exits early.

Smoke tests use [smoke_train.sh](/root/gpufree-data/EgoVLA_Release/training_scripts/robot_finetuning/smoke_train.sh), which disables W&B automatically.

## If Raw Robot Data Changes

If new task folders are downloaded into `data/EgoVLA_SIM`, the processed dataset should be regenerated.

Training does not inspect raw data freshness.
If processed data exists but is stale, training will silently use stale processed data.

This is an easy mistake to make.

## Common Pitfalls

### Loss Looks Like Fresh Training Even Though A Checkpoint Was Used

This can still be normal when:

- `MODEL_PATH` is used with a new `RUN_NAME`
- optimizer state is reset
- scheduler state is reset
- data coverage changed
- task mixture changed

Do not use the first few loss values alone as proof that checkpoint loading failed.

### Processed Data Exists But Does Not Reflect New Raw Tasks

If new raw folders were downloaded, verify whether `EgoVLA_SIM_Processed` was regenerated afterward.

### Resume Confusion

These two are different:

- "resume existing run"
- "start a new run from an older checkpoint"

When a new conversation starts, always identify which one is intended before suggesting commands.

## Minimal Quickstart For A New AI

When dropped into a fresh conversation, check these first:

1. Confirm repo root is `/root/gpufree-data/EgoVLA_Release`.
2. Confirm whether the task is training or simulation eval.
3. If training, use `conda activate vila`.
4. Inspect `training_scripts/robot_finetuning/subset_train_wandb.sh`.
5. Confirm whether user wants exact resume or checkpoint initialization.
6. Confirm current GPU count and compare it with the source checkpoint lineage.
7. Confirm processed data exists under `data/EgoVLA_SIM_Processed`.
8. Confirm intended `RUN_NAME`, `MODEL_PATH`, `N_GPU`, and `GRAD_ACCUM_STEPS`.

If these are clear, the training workflow is usually straightforward.
