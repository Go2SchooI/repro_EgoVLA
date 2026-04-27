
import os
import shutil
import subprocess
import sys
import tqdm
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VILA_ROOT = REPO_ROOT / "VILA"
for path in (str(REPO_ROOT), str(VILA_ROOT)):
  if path not in sys.path:
    sys.path.insert(0, path)


def _maybe_reexec_with_libfix():
  if os.environ.get("EGO_VLA_LIBFIX_DONE") == "1":
    return

  preferred_dirs = [
    "/root/gpufree-data/libfix",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/cuda/lib64",
  ]
  current_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
  prepended_paths = [path for path in preferred_dirs if os.path.isdir(path) and path not in current_paths]

  if not prepended_paths:
    return

  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = ":".join(prepended_paths + current_paths)
  env["EGO_VLA_LIBFIX_DONE"] = "1"
  os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_libfix()


def _prefer_env_boto_stack():
  if os.environ.get("EGO_VLA_PRELOAD_BOTO", "1") != "1":
    return

  try:
    import boto3 as _boto3
    import botocore as _botocore
    import s3transfer as _s3transfer
  except Exception as exc:
    print(f"[WARN] Failed to preload boto stack: {exc}", file=sys.stderr)
    return

  print(
    "[INFO] Preloaded boto stack from env: "
    f"boto3={_boto3.__version__}, "
    f"botocore={_botocore.__version__}, "
    f"s3transfer={_s3transfer.__version__}",
    file=sys.stderr,
  )


_prefer_env_boto_stack()


def _configure_headless_vulkan():
  if os.environ.get("EGO_VLA_HEADLESS_EGL_ICD", "1") != "1":
    return

  xdg_runtime_dir = os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/egovla-xdg-runtime")
  os.makedirs(xdg_runtime_dir, mode=0o700, exist_ok=True)

  icd_path = "/tmp/egovla_nvidia_egl_icd.json"
  icd_contents = """{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.4.312"
  }
}
"""
  if not os.path.exists(icd_path) or open(icd_path, "r").read() != icd_contents:
    with open(icd_path, "w") as icd_file:
      icd_file.write(icd_contents)

  os.environ.setdefault("VK_ICD_FILENAMES", icd_path)
  os.environ.setdefault("DISABLE_LAYER_NV_OPTIMUS_1", "1")
  print(f"[INFO] Using headless Vulkan ICD override: {os.environ['VK_ICD_FILENAMES']}", file=sys.stderr)


_configure_headless_vulkan()


def _canonicalize_cuda_device(device: str) -> str:
  if not isinstance(device, str) or not device.startswith("cuda"):
    return device

  try:
    import torch
  except Exception:
    return device

  try:
    visible_device_count = torch.cuda.device_count()
  except Exception:
    return device

  if visible_device_count <= 0:
    return device

  requested_index = 0
  if ":" in device:
    try:
      requested_index = int(device.split(":", 1)[1])
    except ValueError:
      return device

  if requested_index < visible_device_count:
    return f"cuda:{requested_index}"

  canonical_device = "cuda:0"
  print(
    f"[WARN] Requested device {device}, but only {visible_device_count} CUDA device(s) "
    f"are visible in this process. Falling back to {canonical_device}. "
    f"If you want physical GPU {requested_index}, run with "
    f"CUDA_VISIBLE_DEVICES={requested_index} and EGO_VLA_EVAL_DEVICE=cuda:0.",
    file=sys.stderr,
  )
  return canonical_device


def _convert_video_to_h264(video_path: str) -> None:
  ffmpeg_bin = shutil.which("ffmpeg")
  if ffmpeg_bin is None:
    print(f"[WARN] ffmpeg not found, keeping original video: {video_path}", flush=True)
    return

  source_path = Path(video_path)
  if not source_path.exists():
    print(f"[WARN] Video file missing, skipping H.264 transcode: {video_path}", flush=True)
    return

  temp_output_path = source_path.with_name(f"{source_path.stem}.h264_tmp{source_path.suffix}")
  print(f"h264_transcode_start path={source_path}", flush=True)
  try:
    subprocess.run(
      [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_output_path),
      ],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
      text=True,
    )
  except subprocess.CalledProcessError as exc:
    if temp_output_path.exists():
      temp_output_path.unlink()
    error_tail = exc.stderr.strip().splitlines()[-1] if exc.stderr else str(exc)
    print(
      f"[WARN] Failed to transcode video to H.264, keeping original file: {error_tail}",
      flush=True,
    )
    return

  temp_output_path.replace(source_path)
  print(f"h264_transcode_done path={source_path}", flush=True)

from transformers import HfArgumentParser
from human_plan.vila_train.args import (
  VLATrainingArguments, VLAModelArguments, VLADataArguments
)
from human_plan.ego_bench_eval.rl.config import add_rl_args, build_rl_config

from collections import deque
from omni.isaac.lab.app import AppLauncher

# We fix the seed for tasks to make sure the object position during evaluation
# are never seen during training.
seed_map = {
    "Humanoid-Pour-Balls-v0": 0,
    "Humanoid-Sort-Cans-v0": 1,
    "Humanoid-Insert-Cans-v0": 2,
    "Humanoid-Unload-Cans-v0": 3,
    "Humanoid-Insert-And-Unload-Cans-v0": 4,
    "Humanoid-Push-Box-v0": 5,
    "Humanoid-Open-Drawer-v0": 6,
    "Humanoid-Close-Drawer-v0": 7,
    "Humanoid-Open-Laptop-v0": 8,
    "Humanoid-Flip-Mug-v0": 9,
    "Humanoid-Stack-Can-v0": 10,
    "Humanoid-Stack-Can-Into-Drawer-v0": 11,
}

parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments))
# add argparse arguments
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--room_idx", type=int, default=None, help="Room Idx")
parser.add_argument("--table_idx", type=int, default=None, help="Table Idx")
parser.add_argument("--smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--hand_smooth_weight", type=float, default=None, help="smooth weight")
parser.add_argument("--num_episodes", type=int, default=None, help="episode_label")
parser.add_argument("--num_trials", type=int, default=None, help="trial label")
parser.add_argument("--result_saving_path", type=str, default=None, help="result saving path")
parser.add_argument("--video_saving_path", type=str, default=None, help="video saving path")
parser.add_argument("--save_frames", type=int, default=0, help="result saving path")
parser.add_argument("--project_trajs", type=int, default=0, help="result saving path")
parser.add_argument("--additional_label", type=str, default=None, help="additional_label")
add_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

app_launcher_args, _ = parser.parse_known_args()
app_launcher_args.enable_cameras = True
app_launcher_args.headless = True
app_launcher_args.device = _canonicalize_cuda_device(getattr(app_launcher_args, "device", "cuda:0"))
os.environ["EGO_VLA_EVAL_DEVICE"] = app_launcher_args.device

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

from human_plan.ego_bench_eval.utils import (
    process_input,
    ik_step,
    ik_eval_single_step,
    get_language_instruction
)
import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils import parse_env_cfg
import torch

from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.markers import VisualizationMarkers
# from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.lab.utils.math import subtract_frame_transforms
from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

import cv2
from human_plan.vila_eval.utils.load_model import load_model_eval

def main():

    model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()
    task_args.device = _canonicalize_cuda_device(task_args.device)
    os.environ["EGO_VLA_EVAL_DEVICE"] = task_args.device
    eval_device = torch.device(task_args.device)
    rl_config = build_rl_config(task_args)

    if rl_config.enabled:
      print(f"rl_config {rl_config}", flush=True)
      if rl_config.mode == "offline_rl":
        from human_plan.ego_bench_eval.rl.train_loop import offline_rl_from_config
        offline_rl_from_config(rl_config, eval_device)
        return

    model, tokenizer, model_args, data_args, training_args = load_model_eval(
      model_args, data_args, training_args
    )
    model.to(eval_device)
    if rl_config.enabled and rl_config.freeze_egovla:
      model.eval()
      for param in model.parameters():
        param.requires_grad_(False)

    data_args.sep_query_token = model_args.sep_query_token

    import random
    assert task_args.task in seed_map
    print(f"Setting seed to {seed_map[task_args.task]}")
    random.seed(seed_map[task_args.task])
    randomize_idxes = list(range(10000))
    random.shuffle(randomize_idxes)

    # train data are using idxes 0-49, test start from 50
    set_selection = "Test"
    if set_selection == "Train":
      # curr_random_idx = 0 + task_args.room_idx * task_args.num_trials * task_args.num_episodes
      assert False
    elif set_selection == "Test":
      # We used the first 100 random idxes for training
      # Starting from 101th random ides for evaluation
      curr_random_idx = 100 + (
        task_args.room_idx * 5 + task_args.table_idx
      )  * task_args.num_trials * task_args.num_episodes

    # parse configuration
    env_cfg: BaseEnvCfg = parse_env_cfg(
        task_args.task,
        device=task_args.device,
        num_envs=1,
    )

    fast_scene_mode = os.environ.get("EGO_VLA_EVAL_FAST_SCENE", "0") == "1"
    warmup_steps = int(os.environ.get("EGO_VLA_EVAL_WARMUP_STEPS", "100"))

    env_cfg.episode_length_s = 60 # 60 seconds episode length -> For long horizon tasks
    env_cfg.randomize = not fast_scene_mode
    # create environment
    env_cfg.spawn_background = not fast_scene_mode
    # select background
    room_idx = task_args.room_idx
    table_idx = task_args.table_idx
    env_cfg.room_idx = room_idx
    env_cfg.table_idx = table_idx
    print(
      "eval_scene_config",
      {
        "task": task_args.task,
        "room_idx": room_idx,
        "table_idx": table_idx,
        "randomize": env_cfg.randomize,
        "spawn_background": env_cfg.spawn_background,
        "device": task_args.device,
        "fast_scene_mode": fast_scene_mode,
        "warmup_steps": warmup_steps,
      },
      flush=True,
    )
    print("creating_env", flush=True)
    env: BaseEnv = gym.make(
        task_args.task,
        cfg=env_cfg
    )
    print("env_created", flush=True)
    env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
    print("initial_env_reset_start", flush=True)
    env.reset()
    print("initial_env_reset_done", flush=True)

    # IK controllers
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="pinv")
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    save_path = os.path.join(
      task_args.video_saving_path,
      task_args.additional_label,
      f"inference_{task_args.smooth_weight}_{task_args.hand_smooth_weight}"
    )

    Path(save_path).mkdir(exist_ok=True, parents=True)
    if task_args.result_saving_path is not None:
      result_saving_path = Path(task_args.result_saving_path)
      if str(result_saving_path.parent) not in ("", "."):
        result_saving_path.parent.mkdir(exist_ok=True, parents=True)

    import pickle
    with open("init_poses_fixed_set_100traj.pkl", "rb") as f:
       init_poses = pickle.load(f)

    task_name = task_args.task[9:-3]
    load_name = task_name

    # Collect Initial Hand and EE poses from data -> Only set the arm and hand for start
    from human_plan.ego_bench_eval.utils import TASK_INIT_EPISODE
    episode_list = TASK_INIT_EPISODE[task_name][:task_args.num_episodes]

    hist_len = data_args.predict_future_step * data_args.future_index

    import numpy as np
    cam_intrinsics = np.array([
      [488.6662,   0.0000, 640.0000],
      [  0.0000, 488.6662, 360.0000],
      [  0.0000,   0.0000,   1.0000]
    ])

    padding = 0
    rl_all_successes = []

    rl_enabled = rl_config.enabled
    rl_replay = None
    rl_agent = None
    rl_action_normalizer = None
    rl_actor_obs_normalizer = None
    rl_critic_obs_normalizer = None
    rl_priv_adapter = None
    rl_metrics = {}
    if rl_enabled:
      from human_plan.ego_bench_eval.rl.action_space import (
        ActionNormalizer,
        build_base_chunk,
        denormalize_action,
        pack_action,
        postprocess_action,
        unpack_action,
      )
      from human_plan.ego_bench_eval.rl.features import (
        VectorNormalizer,
        build_actor_obs,
        make_fast_fields,
        make_raw_fields,
      )
      from human_plan.ego_bench_eval.rl.diagnostics import (
        action_diff_rows,
        identity_error_summary,
        summarize_action_diff_rows,
        write_action_diff_logs,
      )
      from human_plan.ego_bench_eval.rl.privileged_state import PrivilegedStateAdapter
      from human_plan.ego_bench_eval.rl.replay_buffer import RLReplayBuffer
      from human_plan.ego_bench_eval.rl.sac import SACRefBC, save_split_checkpoints

      rl_action_normalizer = ActionNormalizer.identity()
      if rl_config.load_rl_checkpoint_path and os.path.exists(rl_config.load_rl_checkpoint_path):
        rl_agent, rl_action_normalizer, rl_ckpt_metadata = SACRefBC.load(
          rl_config.load_rl_checkpoint_path, rl_config, eval_device
        )
        rl_actor_obs_normalizer = VectorNormalizer.from_state(
          rl_ckpt_metadata.get("actor_obs_normalizer")
        )
        rl_critic_obs_normalizer = VectorNormalizer.from_state(
          rl_ckpt_metadata.get("critic_obs_normalizer")
        )
        print(f"rl_loaded_checkpoint path={rl_config.load_rl_checkpoint_path}", flush=True)
      elif rl_config.action_normalizer_path and os.path.exists(rl_config.action_normalizer_path):
        try:
          action_norm_state = torch.load(rl_config.action_normalizer_path, map_location="cpu", weights_only=False)
        except TypeError:
          action_norm_state = torch.load(rl_config.action_normalizer_path, map_location="cpu")
        rl_action_normalizer = ActionNormalizer.from_state(action_norm_state)
      elif rl_config.replay_path and os.path.exists(rl_config.replay_path) and rl_config.mode in (
        "online_rl",
        "eval_rl",
        "eval_identity_actor",
        "eval_tiny_noise",
        "eval_residual_scale_sweep",
      ):
        replay_for_stats = RLReplayBuffer.load(rl_config.replay_path)
        rl_action_normalizer = ActionNormalizer.from_state(
          replay_for_stats.metadata.get("action_normalizer")
        )
        rl_actor_obs_normalizer = VectorNormalizer.from_state(
          replay_for_stats.metadata.get("actor_obs_normalizer")
        )
        rl_critic_obs_normalizer = VectorNormalizer.from_state(
          replay_for_stats.metadata.get("critic_obs_normalizer")
        )
      if rl_config.mode in ("eval_rl", "eval_residual_scale_sweep"):
        if not rl_config.load_rl_checkpoint_path:
          raise ValueError(f"rl.mode={rl_config.mode} requires --rl_load_rl_checkpoint_path")
        if rl_agent is None:
          raise FileNotFoundError(f"RL checkpoint not found: {rl_config.load_rl_checkpoint_path}")
        if rl_actor_obs_normalizer is None:
          raise ValueError(f"rl.mode={rl_config.mode} checkpoint is missing actor_obs_normalizer")

      if rl_config.mode in ("collect_base", "online_rl"):
        if rl_config.replay_path and os.path.exists(rl_config.replay_path) and os.path.getsize(rl_config.replay_path) > 0:
          rl_replay = RLReplayBuffer.load(rl_config.replay_path)
          print(f"rl_loaded_replay path={rl_config.replay_path} size={len(rl_replay)} mode={rl_config.mode}", flush=True)
        else:
          if rl_config.replay_path and os.path.exists(rl_config.replay_path):
            print(f"[WARN] Ignoring empty replay file: {rl_config.replay_path}", flush=True)
          rl_replay = RLReplayBuffer(rl_config.replay_capacity)
      rl_priv_adapter = PrivilegedStateAdapter(env)
      rl_noise_rng = np.random.default_rng(rl_config.noise_seed)
      rl_action_diff_rows = []
      rl_identity_summaries = []

    def _rl_append_transition(prev_ctx, next_ctx, reward, done, success, timeout):
      raw_fields = make_raw_fields(
        h_in=prev_ctx["h_in"],
        h_preout=prev_ctx.get("h_preout"),
        proprio=prev_ctx["proprio"],
        base_chunk=prev_ctx["base_chunk"],
        a_ref=prev_ctx["a_ref"],
        priv_state=prev_ctx["priv_state"],
        a_exec=prev_ctx["a_exec"],
        reward=reward,
        done=done,
        success=success,
        timeout=timeout,
        next_h_in=next_ctx["h_in"],
        next_h_preout=next_ctx.get("h_preout"),
        next_proprio=next_ctx["proprio"],
        next_base_chunk=next_ctx["base_chunk"],
        next_a_ref=next_ctx["a_ref"],
        next_priv_state=next_ctx["priv_state"],
      )
      fast_fields, fast_meta = make_fast_fields(
        raw_fields,
        rl_action_normalizer,
        rl_config.chunk_summary_type,
        rl_config.chunk_summary_steps,
        rl_config.feature_hook,
        rl_actor_obs_normalizer,
        rl_critic_obs_normalizer,
      )
      actor_action_dim = int(prev_ctx["a_ref"].shape[0])
      rl_replay.append(raw_fields, fast_fields)
      rl_replay.metadata.update({
        "action_dim": actor_action_dim,
        "actor_action_dim": actor_action_dim,
        "a_ref_parts": ["left_ee", "right_ee", "left_hand", "right_hand"],
        "action_normalizer": rl_action_normalizer.state_dict(),
        "actor_obs_normalizer": (
          rl_actor_obs_normalizer.state_dict() if rl_actor_obs_normalizer is not None else None
        ),
        "critic_obs_normalizer": (
          rl_critic_obs_normalizer.state_dict() if rl_critic_obs_normalizer is not None else None
        ),
        "chunk_summary_type": rl_config.chunk_summary_type,
        "chunk_summary_steps": tuple(rl_config.chunk_summary_steps),
        "feature_hook": rl_config.feature_hook,
        "fast_field_shapes": fast_meta,
        "privileged_state": rl_priv_adapter.metadata() if rl_priv_adapter is not None else {},
      })
      return raw_fields, fast_fields, fast_meta

    # with torch.inference_mode():
    for episode_idx in episode_list:
      for trial_idx in range(task_args.num_trials):
        print(
          f"episode_trial_start episode={episode_idx[0]} trial={trial_idx} randomize_idx={randomize_idxes[curr_random_idx]}",
          flush=True,
        )
        # seq_name = f"episode_{episode_idx}.hdf5"
        seq_name = episode_idx[0]

        # 30 Hz
        rgb_obs_hist = deque(maxlen=120)
        # original video is 15fps and env is 30 fps
        action_hist_left_ee = deque(maxlen=hist_len)
        action_hist_right_ee = deque(maxlen=hist_len)
        action_hist_left_hand = deque(maxlen=hist_len)
        action_hist_right_hand = deque(maxlen=hist_len)

        seq_save_path = os.path.join(
          save_path,
          task_name,
          f"room_{room_idx}",
          f"table_{table_idx}",
        )
        Path(seq_save_path).mkdir(exist_ok=True, parents=True)
        output_path = os.path.join(
          seq_save_path,
          f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}.mp4"
        )
        if task_args.save_frames:
          frames_output_path = os.path.join(
            seq_save_path,
            f"{task_name}_room_{room_idx}_table_{table_idx}_episode_{episode_idx}_{trial_idx}"
          )
          Path(frames_output_path).mkdir(exist_ok=True, parents=True)
        fps = 15
        out = cv2.VideoWriter(
          output_path, 
          #  seq_save_path,
          cv2.VideoWriter_fourcc(*"mp4v"), 
          fps, (1280, 720)
        )

        # def init_env():
        if True:
            # reset
          curr_random_idx += 1
          env.cfg.randomize_idx = randomize_idxes[curr_random_idx]
          print(
            f"rollout_env_reset_start episode={episode_idx[0]} trial={trial_idx} randomize_idx={env.cfg.randomize_idx}",
            flush=True,
          )
          env_results = env.reset()
          print(
            f"rollout_env_reset_done episode={episode_idx[0]} trial={trial_idx}",
            flush=True,
          )
          left_ik_controller.reset()
          right_ik_controller.reset()
          padding_idx = padding
          # for padding_idx in range(padding):

          left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
          right_dof = init_poses[load_name][seq_name][padding]["right_dof"]

          for idx in range(warmup_steps):
            if idx % max(1, min(10, warmup_steps)) == 0:
              print(
                f"warmup_progress episode={episode_idx[0]} trial={trial_idx} step={idx}/{warmup_steps}",
                flush=True,
              )
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_dof = init_poses[load_name][seq_name][padding]["left_dof"]
            right_dof = init_poses[load_name][seq_name][padding]["right_dof"]
            
            left_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["left_ee"]
            right_ee_pose_traj_gt = init_poses[load_name][seq_name][padding]["right_ee"]

            ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world, 
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              left_ee_pose_traj_gt, right_ee_pose_traj_gt,
              left_dof, right_dof,
              action
            )
            env_results = env.step(action)
            rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            rgb_obs = cv2.resize(rgb_obs, (384, 384))
          print(
            f"warmup_done episode={episode_idx[0]} trial={trial_idx}",
            flush=True,
          )
        rgb_obs_hist.append(rgb_obs)
        count = padding

        result = False
        rl_pending_transition = None
        rl_episode_success_values = []
        from human_plan.ego_bench_eval.utils import TASK_MAX_HORIZON
        max_horizon = TASK_MAX_HORIZON[task_args.task]
        print(
          f"rollout_start episode={episode_idx[0]} trial={trial_idx} max_horizon={max_horizon}",
          flush=True,
        )

        for i in tqdm.tqdm(range(max_horizon)):
          # run everything in inference mode
          # obtain quantities from simulation
          rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]

          from human_plan.ego_bench_eval.utils import process_proprio_input

          proprio_input, raw_proprio_inputs = process_proprio_input(
            env_results[0]["left_finger_tip_pos"].cpu().numpy(),
            env_results[0]["right_finger_tip_pos"].cpu().numpy(),
            env_results[0]["left_ee_pose"].cpu().numpy(),
            env_results[0]["right_ee_pose"].cpu().numpy(),
            env_results[0]["qpos"],
            cam_intrinsics,
            input_hand_dof=data_args.input_hand_dof
          )

          rgb_obs = cv2.resize(rgb_obs, (384, 384))
          rgb_obs_hist.append(rgb_obs)

          raw_language_instruction = get_language_instruction(
              task_args.task
          )

          raw_data_dict = process_input(
              rgb_obs_hist, 
              proprio_input.to(eval_device),
              raw_language_instruction,
              data_args, model_args, tokenizer
          )

          raw_data_dict.update(raw_proprio_inputs)
          with torch.inference_mode():
            if rl_enabled:
              action_dict = ik_eval_single_step(
                  raw_data_dict,
                  model, tokenizer,
                  return_rl_features=True,
              )
            else:
              action_dict = ik_eval_single_step(
                  raw_data_dict,
                  model, tokenizer,
              )

          from human_plan.ego_bench_eval.utils import smooth_action, repeat_action
          repeated_right_ee = repeat_action(action_dict["right_ee_pose"], data_args.future_index)
          repeated_left_ee = repeat_action(action_dict["left_ee_pose"], data_args.future_index)
          repeated_left_hand = repeat_action(action_dict["left_qpos_multi_step"], data_args.future_index)
          repeated_right_hand = repeat_action(action_dict["right_qpos_multi_step"], data_args.future_index)
          if rl_enabled:
            assert repeated_right_ee.shape[0] == hist_len
            assert repeated_left_ee.shape[0] == hist_len
            assert repeated_left_hand.shape[0] == hist_len
            assert repeated_right_hand.shape[0] == hist_len
          action_hist_right_ee.append(repeated_right_ee)
          action_hist_left_ee.append(repeated_left_ee)

          action_hist_left_hand.append(repeated_left_hand)
          action_hist_right_hand.append(repeated_right_hand)

          action_left_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_left_ee
          )

          action_right_ee = smooth_action(
            hist_len, task_args.smooth_weight, action_hist_right_ee
          )

          action_left_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_left_hand
          )
          action_right_hand = smooth_action(
            hist_len, task_args.hand_smooth_weight, action_hist_right_hand
          )

          if rl_enabled:
            assert action_left_ee.shape == (7,)
            assert action_right_ee.shape == (7,)
            assert action_left_hand.shape == (12,)
            assert action_right_hand.shape == (12,)
            base_chunk = build_base_chunk(action_dict)
            a_ref = pack_action(
              action_left_ee,
              action_right_ee,
              action_left_hand,
              action_right_hand,
            )
            rl_features = action_dict.get("rl_features", {})
            h_in = rl_features.get("h_in")
            h_preout = rl_features.get("h_preout")
            if h_in is None:
              raise RuntimeError("RL enabled but h_in feature hook is missing")
            actor_feature = h_in
            if rl_config.feature_hook == "pre_output":
              if h_preout is None:
                raise RuntimeError("rl.feature_hook=pre_output but h_preout is missing")
              actor_feature = h_preout
            actor_obs, actor_obs_meta = build_actor_obs(
              actor_feature,
              raw_data_dict["proprio_input"],
              base_chunk,
              a_ref,
              rl_action_normalizer,
              rl_config.chunk_summary_type,
              rl_config.chunk_summary_steps,
            )
            if rl_actor_obs_normalizer is not None:
              actor_obs = rl_actor_obs_normalizer.normalize(actor_obs)
            priv_state, priv_meta = rl_priv_adapter.build(env_results[0])
            rl_curr_ctx = {
              "h_in": h_in,
              "h_preout": h_preout,
              "proprio": raw_data_dict["proprio_input"],
              "base_chunk": base_chunk,
              "a_ref": a_ref,
              "actor_obs": actor_obs,
              "actor_obs_meta": actor_obs_meta,
              "priv_state": priv_state,
              "priv_meta": priv_meta,
            }
            if rl_replay is not None and rl_pending_transition is not None:
              _rl_append_transition(
                rl_pending_transition["ctx"],
                rl_curr_ctx,
                rl_pending_transition["reward"],
                rl_pending_transition["done"],
                rl_pending_transition["success"],
                rl_pending_transition["timeout"],
              )
              rl_pending_transition = None

            a_ref_norm = rl_action_normalizer.normalize(a_ref)
            a_actor_for_log = None
            a_actor_norm = None
            if rl_config.mode in ("eval_rl", "online_rl", "eval_residual_scale_sweep"):
              if rl_agent is None:
                if rl_config.mode in ("eval_rl", "eval_residual_scale_sweep"):
                  raise RuntimeError(f"{rl_config.mode} must use a loaded RL checkpoint")
                rl_agent = SACRefBC(
                  actor_obs_dim=actor_obs.shape[0],
                  critic_obs_dim=priv_state.shape[0],
                  action_dim=a_ref.shape[0],
                  cfg=rl_config,
                  device=eval_device,
                  ref_obs_slice=actor_obs_meta.get("ref_obs_slice"),
                )
                print(
                  f"rl_initialized_random actor_obs_dim={actor_obs.shape[0]} critic_obs_dim={priv_state.shape[0]} actor_action_dim={a_ref.shape[0]}",
                  flush=True,
                )
              deterministic = rl_config.mode in ("eval_rl", "eval_residual_scale_sweep") and rl_config.deterministic_eval
              a_actor_norm = rl_agent.act(actor_obs, deterministic=deterministic)
              a_actor_norm = np.clip(
                a_actor_norm,
                -float(rl_config.action_norm_clip),
                float(rl_config.action_norm_clip),
              )
              a_actor_for_log = postprocess_action(
                denormalize_action(a_actor_norm, rl_action_normalizer),
                a_ref,
                env=env,
              )
              if rl_config.mode == "eval_residual_scale_sweep":
                a_exec_norm = a_ref_norm + float(rl_config.residual_scale) * (a_actor_norm - a_ref_norm)
                a_exec_norm = np.clip(
                  a_exec_norm,
                  -float(rl_config.action_norm_clip),
                  float(rl_config.action_norm_clip),
                )
              else:
                a_exec_norm = a_actor_norm
              a_exec = denormalize_action(a_exec_norm, rl_action_normalizer)
              a_exec = postprocess_action(a_exec, a_ref, env=env)
            elif rl_config.mode == "eval_identity_actor":
              a_exec_norm = a_ref_norm.copy()
              a_exec = denormalize_action(a_exec_norm, rl_action_normalizer)
              a_exec = postprocess_action(a_exec, a_ref, env=env)
              identity_summary = identity_error_summary(a_ref, a_exec)
              rl_identity_summaries.append(identity_summary)
              if i == 0:
                print(
                  "identity_actor_step "
                  f"identity_max_abs_error={identity_summary['identity_max_abs_error']:.8g} "
                  f"identity_mean_abs_error={identity_summary['identity_mean_abs_error']:.8g} "
                  f"per_group_identity_error={identity_summary['per_group_identity_error']}",
                  flush=True,
                )
            elif rl_config.mode == "eval_tiny_noise":
              if rl_config.noise_type == "uniform":
                epsilon = rl_noise_rng.uniform(-1.0, 1.0, size=a_ref_norm.shape).astype(np.float32)
              else:
                epsilon = rl_noise_rng.standard_normal(size=a_ref_norm.shape).astype(np.float32)
              a_exec_norm = a_ref_norm + float(rl_config.noise_scale) * epsilon
              a_exec_norm = np.clip(
                a_exec_norm,
                -float(rl_config.action_norm_clip),
                float(rl_config.action_norm_clip),
              )
              a_exec = denormalize_action(a_exec_norm, rl_action_normalizer)
              a_exec = postprocess_action(a_exec, a_ref, env=env)
            else:
              a_exec = a_ref.copy()
            a_exec_parts = unpack_action(a_exec)
            action_left_ee = a_exec_parts["left_ee"]
            action_right_ee = a_exec_parts["right_ee"]
            action_left_hand = a_exec_parts["left_hand"]
            action_right_hand = a_exec_parts["right_hand"]
            rl_curr_ctx["a_exec"] = a_exec
            rl_curr_ctx["a_ref_norm"] = a_ref_norm
            rl_curr_ctx["a_exec_norm"] = rl_action_normalizer.normalize(a_exec)
            if a_actor_norm is not None:
              rl_curr_ctx["a_actor_norm"] = a_actor_norm
            if a_actor_for_log is not None:
              rl_curr_ctx["a_actor"] = a_actor_for_log
            if rl_config.mode in (
              "eval_rl",
              "eval_identity_actor",
              "eval_tiny_noise",
              "eval_residual_scale_sweep",
            ):
              rl_action_diff_rows.extend(action_diff_rows(
                step=i,
                a_ref=a_ref,
                a_exec=a_exec,
                a_actor=a_actor_for_log,
                method=rl_config.mode,
                episode=episode_idx[0],
                trial=trial_idx,
                room_idx=room_idx,
                table_idx=table_idx,
                checkpoint=rl_config.load_rl_checkpoint_path,
                residual_scale=(
                  float(rl_config.residual_scale)
                  if rl_config.mode == "eval_residual_scale_sweep"
                  else (1.0 if rl_config.mode == "eval_rl" else None)
                ),
                noise_scale=float(rl_config.noise_scale) if rl_config.mode == "eval_tiny_noise" else None,
              ))
            if rl_config.debug_dump_shapes or rl_config.mode == "debug_trace_action_path":
              rl_curr_ctx["debug_shapes"] = {
                "raw_pred.shape": tuple(np.asarray(action_dict["pred"]).shape),
                "h_in.shape": tuple(np.asarray(h_in).shape),
                "h_preout.shape": tuple(np.asarray(h_preout).shape) if h_preout is not None else None,
                "base_chunk.shape": tuple(base_chunk.shape),
                "action_dict_shapes": {
                  key: tuple(np.asarray(action_dict[key]).shape)
                  for key in (
                    "left_ee_pose",
                    "right_ee_pose",
                    "left_qpos_multi_step",
                    "right_qpos_multi_step",
                  )
                },
                "repeated_shapes": {
                  "left_ee": tuple(repeated_left_ee.shape),
                  "right_ee": tuple(repeated_right_ee.shape),
                  "left_hand": tuple(repeated_left_hand.shape),
                  "right_hand": tuple(repeated_right_hand.shape),
                },
                "smooth_action_shapes": {
                  "left_ee": tuple(action_left_ee.shape),
                  "right_ee": tuple(action_right_ee.shape),
                  "left_hand": tuple(action_left_hand.shape),
                  "right_hand": tuple(action_right_hand.shape),
                },
                "packed_a_ref.shape": tuple(a_ref.shape),
                "a_exec_matches_a_ref": bool(np.allclose(a_exec, a_ref)),
                "a_exec_postprocessed": bool(rl_config.mode in (
                  "eval_rl",
                  "online_rl",
                  "eval_identity_actor",
                  "eval_tiny_noise",
                  "eval_residual_scale_sweep",
                )),
                "a_ref_norm.shape": tuple(a_ref_norm.shape),
                "a_exec_norm.shape": tuple(rl_curr_ctx["a_exec_norm"].shape),
                "actor_obs.shape": tuple(actor_obs.shape),
                "priv_state.shape": tuple(priv_state.shape),
                "base_chunk_summary.shape": tuple(actor_obs_meta["base_chunk_summary_shape"]),
                "critic_obs_keys": priv_meta["keys"],
                "critic_obs_schema_hash": priv_meta["schema_hash"],
                "env_num_actions": int(env.num_actions),
              }

          ik_step(
              env,
              left_ik_controller,
              right_ik_controller,

              left_ik_commands_world,
              right_ik_commands_world,
              
              left_ik_commands_robot,
              right_ik_commands_robot,

              action_left_ee,
              action_right_ee,

              action_left_hand,
              action_right_hand,

              action
          )
          if rl_enabled:
            assert action.shape == (env.scene.num_envs, env.num_actions), (
              f"env action tensor shape {tuple(action.shape)} expected {(env.scene.num_envs, env.num_actions)}"
            )
          env_results = env.step(action)

          if rl_enabled:
            success = env_results[0]["success"].sum().item() == 1
            timeout = (i + 1) >= max_horizon
            reward = 1.0 if success else 0.0
            done = bool(success or timeout)
            rl_episode_success_values.append(float(success))
            if rl_replay is not None:
              if done:
                _rl_append_transition(
                  rl_curr_ctx,
                  rl_curr_ctx,
                  reward,
                  done,
                  success,
                  timeout,
                )
              else:
                rl_pending_transition = {
                  "ctx": rl_curr_ctx,
                  "reward": reward,
                  "done": done,
                  "success": success,
                  "timeout": timeout,
                }

            if rl_config.mode == "online_rl" and rl_replay is not None and len(rl_replay) >= rl_config.min_replay_size:
              if rl_agent is None:
                raise RuntimeError("online_rl has replay samples but no RL agent")
              for _ in range(rl_config.updates_per_env_step):
                batch = rl_replay.sample_batch(rl_config.batch_size, eval_device)
                rl_metrics = rl_agent.update(batch)
              if i % 25 == 0:
                success_rate = float(np.mean(rl_episode_success_values)) if rl_episode_success_values else 0.0
                print(
                  "online_rl_update "
                  + " ".join(f"{key}={value:.5g}" for key, value in rl_metrics.items())
                  + f" success_rate={success_rate:.5g} replay_size={len(rl_replay)}",
                  flush=True,
                )

            if rl_config.mode == "debug_trace_action_path":
              debug_raw = make_raw_fields(
                h_in=rl_curr_ctx["h_in"],
                h_preout=rl_curr_ctx.get("h_preout"),
                proprio=rl_curr_ctx["proprio"],
                base_chunk=rl_curr_ctx["base_chunk"],
                a_ref=rl_curr_ctx["a_ref"],
                priv_state=rl_curr_ctx["priv_state"],
                a_exec=rl_curr_ctx["a_exec"],
                reward=reward,
                done=done,
                success=success,
                timeout=timeout,
                next_h_in=rl_curr_ctx["h_in"],
                next_h_preout=rl_curr_ctx.get("h_preout"),
                next_proprio=rl_curr_ctx["proprio"],
                next_base_chunk=rl_curr_ctx["base_chunk"],
                next_a_ref=rl_curr_ctx["a_ref"],
                next_priv_state=rl_curr_ctx["priv_state"],
              )
              debug_fast, debug_meta = make_fast_fields(
                debug_raw,
                rl_action_normalizer,
                rl_config.chunk_summary_type,
                rl_config.chunk_summary_steps,
                rl_config.feature_hook,
                rl_actor_obs_normalizer,
                rl_critic_obs_normalizer,
              )
              debug_shapes = dict(rl_curr_ctx.get("debug_shapes", {}))
              debug_shapes["env_action_tensor.shape"] = tuple(action.shape)
              debug_shapes["success"] = bool(success)
              debug_shapes["timeout"] = bool(timeout)
              debug_path = Path(rl_config.save_debug_transition_path)
              if str(debug_path.parent) not in ("", "."):
                debug_path.parent.mkdir(parents=True, exist_ok=True)
              torch.save(
                {
                  "shapes": debug_shapes,
                  "raw": debug_raw,
                  "fast": debug_fast,
                  "fast_meta": debug_meta,
                  "privileged_state_meta": rl_curr_ctx["priv_meta"],
                },
                debug_path,
              )
              print(f"rl_debug_trace_saved path={rl_config.save_debug_transition_path} shapes={debug_shapes}", flush=True)
              result = bool(success)
              break

          # Success 
          if env_results[0]["success"].sum().item() == 1:
            result = True
            break

          result_img_3d = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, ::-1].copy()
          if task_args.project_trajs == 1:
            from human_plan.utils.visualization import (
              project_points
            )

            pred_3d = action_dict["pred_3d"]
            proj_2d = project_points(
              pred_3d, cam_intrinsics
            )
            proj_2d = proj_2d.reshape(-1, 2, 2)

            for fi in range(proj_2d.shape[0]-1):
              for j in range(2):
                result_img_3d = cv2.circle(
                  result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  5, (0, 255, 0), thickness=-1
                )
                if fi < proj_2d.shape[0] - 1:
                  result_img_3d = cv2.line(
                    result_img_3d, 
                  (int(proj_2d[fi, j, 0]),int(proj_2d[fi, j, 1])),
                  (int(proj_2d[fi + 1, j, 0]),int(proj_2d[fi + 1, j, 1])),
                    (0, 255, 0), thickness=2
                  )  

          out.write(result_img_3d)

          if task_args.save_frames:
            cv2.imwrite(
              os.path.join(frames_output_path, f"{i}.jpg"),
              result_img_3d
            )
          count += 1

        if task_args.result_saving_path is not None:
          with open(task_args.result_saving_path, "a") as f:
            f.write(f"Task: {task_name}, Room Idx: {room_idx}, Table Idx: {table_idx}, Episode Label: {episode_idx[0]}, Trial Label: {trial_idx}, Result: {result} \n")
            subtask_string = ""
            for key in env_results[0].keys():
              if "success" in key:
                subtask_string += f"{key}: {env_results[0][key].sum().item()} "
            subtask_string += "\n"
            f.write(subtask_string)
        if rl_enabled:
          rl_all_successes.append(float(result))
        
        out.release()
        _convert_video_to_h264(output_path)
        # close the simulator
    if rl_enabled:
      success_rate = float(np.mean(rl_all_successes)) if rl_all_successes else 0.0
      print(f"rl_rollout_done mode={rl_config.mode} success_rate={success_rate:.5g}", flush=True)
      if rl_action_diff_rows:
        diff_summary = summarize_action_diff_rows(
          rl_action_diff_rows,
          step_start=rl_config.action_diff_step_start,
          step_end=rl_config.action_diff_step_end,
        )
        print(
          "rl_action_diff_summary "
          f"mode={rl_config.mode} "
          f"step_start={rl_config.action_diff_step_start} "
          f"step_end={rl_config.action_diff_step_end} "
          f"summary={diff_summary}",
          flush=True,
        )
        if rl_config.action_diff_log_path:
          write_action_diff_logs(
            rl_config.action_diff_log_path,
            rl_action_diff_rows,
            metadata={
              "mode": rl_config.mode,
              "task": task_args.task,
              "room_idx": room_idx,
              "table_idx": table_idx,
              "checkpoint": rl_config.load_rl_checkpoint_path,
              "noise_scale": rl_config.noise_scale,
              "noise_type": rl_config.noise_type,
              "noise_seed": rl_config.noise_seed,
              "residual_scale": rl_config.residual_scale,
              "success_rate": success_rate,
            },
          )
          print(f"rl_action_diff_saved path={rl_config.action_diff_log_path}", flush=True)
      if rl_config.mode == "eval_identity_actor" and rl_identity_summaries:
        max_error = max(item["identity_max_abs_error"] for item in rl_identity_summaries)
        mean_error = float(np.mean([item["identity_mean_abs_error"] for item in rl_identity_summaries]))
        per_group = {}
        for group_name in rl_identity_summaries[0]["per_group_identity_error"].keys():
          per_group[group_name] = {
            "mean_abs": float(np.mean([
              item["per_group_identity_error"][group_name]["mean_abs"]
              for item in rl_identity_summaries
            ])),
            "max_abs": float(np.max([
              item["per_group_identity_error"][group_name]["max_abs"]
              for item in rl_identity_summaries
            ])),
          }
        print(
          "identity_actor_summary "
          f"identity_max_abs_error={max_error:.8g} "
          f"identity_mean_abs_error={mean_error:.8g} "
          f"per_group_identity_error={per_group}",
          flush=True,
        )
      if rl_replay is not None and len(rl_replay) > 0:
        if rl_config.mode == "collect_base":
          rl_action_normalizer = rl_replay.fit_action_normalizer(clip=rl_config.action_norm_clip)
          rl_actor_obs_normalizer, rl_critic_obs_normalizer = rl_replay.fit_obs_normalizers(
            rl_action_normalizer,
            rl_config.chunk_summary_type,
            rl_config.chunk_summary_steps,
            rl_config.feature_hook,
            clip=rl_config.obs_norm_clip,
          )
          rl_replay.rebuild_fast_fields(
            rl_action_normalizer,
            rl_config.chunk_summary_type,
            rl_config.chunk_summary_steps,
            rl_config.feature_hook,
            rl_actor_obs_normalizer,
            rl_critic_obs_normalizer,
          )
          if rl_config.action_normalizer_path:
            action_norm_path = Path(rl_config.action_normalizer_path)
            if str(action_norm_path.parent) not in ("", "."):
              action_norm_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rl_action_normalizer.state_dict(), rl_config.action_normalizer_path)
        rl_replay.save(rl_config.replay_path)
        print(f"rl_replay_saved path={rl_config.replay_path} size={len(rl_replay)}", flush=True)
      if rl_agent is not None and rl_config.mode in ("online_rl", "eval_rl"):
        checkpoint_path = rl_config.save_rl_checkpoint_path
        if checkpoint_path and rl_config.mode == "online_rl":
          rl_agent.save(
            checkpoint_path,
            action_normalizer=rl_action_normalizer,
            actor_obs_normalizer=rl_actor_obs_normalizer,
            critic_obs_normalizer=rl_critic_obs_normalizer,
            metadata={
              "success_rate": success_rate,
              "replay_path": rl_config.replay_path,
            },
          )
          save_split_checkpoints(rl_agent, rl_config.actor_checkpoint_path, rl_config.critic_checkpoint_path)
          print(f"rl_checkpoint_saved path={checkpoint_path}", flush=True)
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
