#!/usr/bin/env bash
set -euo pipefail

TASK=${1:?Missing task name}
ROOM_IDX=${2:?Missing room index}
TABLE_IDX=${3:?Missing table index}
SMOOTH_WEIGHT=${4:?Missing ee smoothing weight}
NUM_EPISODES=${5:?Missing episode count}
NUM_TRIALS=${6:?Missing trial count}
SAVING_PATH=${7:?Missing result saving path}
SAVE_FRAMES=${8:?Missing save_frames flag}
PROJ_TRAJS=${9:?Missing project_trajs flag}
HAND_SMOOTH_WEIGHT=${10:?Missing hand smoothing weight}
video_saving_path=${11:?Missing video output path}
additional_label=${12:?Missing evaluation tag}
use_per_step_instruction=${13:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DEFAULT_CHECKPOINT="/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v5-from14000/checkpoint-3500"
CHECKPOINT_PATH="${EGO_VLA_CHECKPOINT_PATH:-$DEFAULT_CHECKPOINT}"
PYTHON_BIN="${EGO_VLA_EVAL_PYTHON:-python}"
EVAL_DEVICE="${EGO_VLA_EVAL_DEVICE:-cuda:0}"

if [ -n "${EGO_VLA_SETUP_SCRIPT:-}" ]; then
  if [ ! -f "$EGO_VLA_SETUP_SCRIPT" ]; then
    echo "Setup script not found: $EGO_VLA_SETUP_SCRIPT" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$EGO_VLA_SETUP_SCRIPT"
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/VILA${PYTHONPATH:+:$PYTHONPATH}"

LIBFIX_DIR="${EGO_VLA_LIBFIX_DIR:-/root/gpufree-data/libfix}"
if [ -d "$LIBFIX_DIR" ]; then
  export LD_LIBRARY_PATH="$LIBFIX_DIR:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint directory not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

LEGACY_MODEL_BASE="$REPO_ROOT/checkpoints/ego_vla_checkpoint"
if [ ! -d "$LEGACY_MODEL_BASE" ]; then
  echo "Warning: legacy model base not found at $LEGACY_MODEL_BASE" >&2
  echo "If model loading fails, check human_plan/vila_eval/utils/load_model.py." >&2
fi

LOG_ROOT=logs

RUN_NAME=temp
OUTPUT_DIR=$LOG_ROOT/$RUN_NAME

bs=16
echo "Using checkpoint: $CHECKPOINT_PATH"

# deepspeed human_plan/train/train_vla_finetune_llava.py \
"$PYTHON_BIN" human_plan/ego_bench_eval/ik_agent_30hz.py \
    --model_name_or_path "$CHECKPOINT_PATH" \
    --device "$EVAL_DEVICE" \
    --headless \
    --enable_cameras \
    --version qwen2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --group_by_modality_length False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 100 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --gradient_accumulation_steps 1 \
    --eval_data_mixture otv_sim_fixed_set_aug_AUG_SHIFT_30Hz_train_sub100 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --future_index 1 \
    --predict_future_step 30 \
    --max_action 1 \
    --min_action 0 \
    --add_his_obs_step 5 \
    --add_his_imgs True \
    --add_his_img_skip 6 \
    --num_action_bins 256 \
    --action_tokenizer uniform \
    --invalid_token_weight 0.1 \
    --mask_input True \
    --add_current_language_description False \
    --traj_decoder_type transformer_split_action_v2 \
    --raw_action_label True \
    --traj_action_output_dim 48 \
    --input_placeholder_diff_index True \
    --ee_loss_coeff 20.0 \
    --hand_loss_coeff 5.0 \
    --hand_loss_dim 6 \
    --ee_2d_loss_coeff 0.0 \
    --ee_rot_loss_coeff 5.0 \
    --hand_kp_loss_coeff 0.0 \
    --next_token_loss_coeff 0.0 \
    --traj_action_output_ee_2d_dim 0 \
    --traj_action_output_ee_dim 6 \
    --traj_action_output_hand_dim 30  \
    --traj_action_output_ee_rot_dim 12 \
    --ee_rot_representation rot6d \
    --correct_transformation True \
    --include_2d_label True \
    --include_rot_label True \
    --use_short_language_label True \
    --no_norm_ee_label True \
    --lazy_preprocess True \
    --tf32 True \
    --merge_hand True \
    --use_mano True \
    --sep_proprio True \
    --sep_query_token True \
    --loss_use_l1 True \
    --task "$TASK" \
    --room_idx "$ROOM_IDX" \
    --table_idx "$TABLE_IDX" \
    --smooth_weight "$SMOOTH_WEIGHT" \
    --num_episodes "$NUM_EPISODES" \
    --num_trials "$NUM_TRIALS" \
    --result_saving_path "$SAVING_PATH" \
    --save_frames "$SAVE_FRAMES" \
    --project_trajs "$PROJ_TRAJS" \
    --hand_smooth_weight "$HAND_SMOOTH_WEIGHT" \
    --video_saving_path "$video_saving_path" \
    --additional_label "$additional_label" \
    "${@:14}"
