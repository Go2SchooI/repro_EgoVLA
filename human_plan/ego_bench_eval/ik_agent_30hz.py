
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

    model, tokenizer, model_args, data_args, training_args = load_model_eval(
      model_args, data_args, training_args
    )
    model.to(eval_device)

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
            action_dict = ik_eval_single_step(
                raw_data_dict,
                model, tokenizer,
            )

          from human_plan.ego_bench_eval.utils import smooth_action, repeat_action
          action_hist_right_ee.append(
            repeat_action(action_dict["right_ee_pose"], data_args.future_index)
          )
          action_hist_left_ee.append(
            repeat_action(action_dict["left_ee_pose"], data_args.future_index)
          )

          action_hist_left_hand.append(
            repeat_action(action_dict["left_qpos_multi_step"], data_args.future_index)
          )
          action_hist_right_hand.append(
            repeat_action(action_dict["right_qpos_multi_step"], data_args.future_index)
          )

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
          env_results = env.step(action)

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
        
        out.release()
        _convert_video_to_h264(output_path)
        # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
