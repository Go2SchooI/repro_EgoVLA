cd /root/gpufree-data/EgoVLA_Release
conda activate vila
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
N_GPU=6 \
PER_DEVICE_BS=1 \
GRAD_ACCUM_STEPS=3 \
NUM_EPOCHS=6 \
RUN_NAME=otv-fixed-set-subset-6gpu-wandb-v5-from14000 \
MODEL_PATH=/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-5gpu-wandb-v4-from2000/checkpoint-14000 \
bash training_scripts/robot_finetuning/subset_train_wandb.sh

cd /root/gpufree-data/EgoVLA_Release
conda activate egovla-sim
EGO_VLA_EVAL_DEVICE=cuda:0 EGO_VLA_SIGLIP_ATTN_IMPLEMENTATION=sdpa bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh Humanoid-Open-Laptop-v0 1 3 0.2 3 1 result_log.txt 0 0 0.8 video_output evaluation_tag


Humanoid-Push-Box-v0
Humanoid-Open-Drawer-v0
Humanoid-Close-Drawer-v0
Humanoid-Pour-Balls-v0
Humanoid-Flip-Mug-v0
Humanoid-Open-Laptop-v0
Humanoid-Stack-Can-v0
Humanoid-Unload-Cans-v0
Humanoid-Insert-Cans-v0
Humanoid-Stack-Can-Into-Drawer-v0
Humanoid-Sort-Cans-v0
Humanoid-Insert-And-Unload-Cans-v0


source /opt/conda/etc/profile.d/conda.sh
conda activate egovla-sim

cd /root/gpufree-data/EgoVLA_Release

export EGO_VLA_EVAL_PYTHON=/opt/conda/envs/egovla-sim/bin/python
export EGO_VLA_CHECKPOINT_PATH=/root/gpufree-data/EgoVLA_Release/checkpoints/otv-fixed-set-subset-6gpu-wandb-v5-from14000/checkpoint-3500

unset EGO_VLA_EVAL_FAST_SCENE
export EGO_VLA_EVAL_FAST_SCENE=1

export EGO_VLA_EVAL_WARMUP_STEPS=100

for ROOM in 2 3; do
  for TABLE in 1 2 3; do
    bash human_plan/ego_bench_eval/fullpretrain_p30_h5_transv2.sh \
      Humanoid-Open-Laptop-v0 $ROOM $TABLE 0.2 6 1 \
      rl_runs/open_laptop_smooth02/collect_result.txt \
      0 0 0.2 \
      rl_runs/open_laptop_smooth02/videos collect_base 0 \
      --rl_enabled true \
      --rl_mode collect_base \
      --rl_replay_path rl_runs/open_laptop_smooth02/base_replay.pt \
      --rl_action_normalizer_path rl_runs/open_laptop_smooth02/action_norm.pt
  done
done
