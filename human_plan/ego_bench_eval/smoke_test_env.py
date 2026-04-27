import os
import sys
import argparse
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
    f"CUDA_VISIBLE_DEVICES={requested_index} and --device cuda:0.",
    file=sys.stderr,
  )
  return canonical_device

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Minimal IsaacLab smoke test for EgoVLA benchmark tasks.")
parser.add_argument("--task", type=str, default="Humanoid-Push-Box-v0", help="Task name registered in the benchmark.")
parser.add_argument("--room_idx", type=int, default=1, help="Background room index.")
parser.add_argument("--table_idx", type=int, default=1, help="Table index.")
parser.add_argument("--num_steps", type=int, default=2, help="Number of zero-action steps to run.")
parser.add_argument(
  "--full-scene",
  action="store_true",
  help="Load the benchmark background room and default high-resolution cameras. Slower but closer to full eval.",
)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()
args.enable_cameras = True
args.device = _canonicalize_cuda_device(getattr(args, "device", "cuda:0"))
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _configure_smoke_test_cfg(env_cfg: object) -> None:
  if args.full_scene:
    env_cfg.randomize = True
    env_cfg.spawn_background = True
    return

  # Fast smoke test: keep the benchmark task and cameras, but skip the heavy room assets
  # and reduce render cost so reset() can finish in a reasonable amount of time.
  env_cfg.randomize = False
  env_cfg.spawn_background = False
  env_cfg.left_eye_camera.width = 160
  env_cfg.left_eye_camera.height = 90
  env_cfg.right_eye_camera.width = 160
  env_cfg.right_eye_camera.height = 90
  env_cfg.main_camera.width = 320
  env_cfg.main_camera.height = 180
  env_cfg.left_hand_camera.width = 320
  env_cfg.left_hand_camera.height = 180
  env_cfg.right_hand_camera.width = 320
  env_cfg.right_hand_camera.height = 180


def main():
  import gymnasium as gym
  import torch

  from omni.isaac.lab_tasks.utils import parse_env_cfg
  from humanoid.tasks.base_env import BaseEnv, BaseEnvCfg

  env_cfg: BaseEnvCfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
  env_cfg.episode_length_s = 60
  env_cfg.room_idx = args.room_idx
  env_cfg.table_idx = args.table_idx
  _configure_smoke_test_cfg(env_cfg)

  print("smoke_test_mode", "full_scene" if args.full_scene else "fast")
  print("spawn_background", env_cfg.spawn_background)
  print("randomize", env_cfg.randomize)
  print("fixed_camera_shape", (env_cfg.main_camera.height, env_cfg.main_camera.width))

  env: BaseEnv = gym.make(args.task, cfg=env_cfg)
  try:
    env_results = env.reset()
    action = torch.zeros((env.scene.num_envs, env.num_actions), device=env.robot.device)

    for _ in range(args.num_steps):
      env_results = env.step(action)

    observation = env_results[0]
    required_keys = ["fixed_rgb", "qpos", "success"]
    missing_keys = [key for key in required_keys if key not in observation]
    if missing_keys:
      raise RuntimeError(f"Missing observation keys: {missing_keys}")

    print("task", args.task)
    print("room_idx", args.room_idx)
    print("table_idx", args.table_idx)
    print("num_actions", env.num_actions)
    print("fixed_rgb_shape", tuple(observation["fixed_rgb"].shape))
    print("qpos_shape", tuple(observation["qpos"].shape))
    print("success_shape", tuple(observation["success"].shape))
    print("smoke_test_ok", True)
  finally:
    env.close()


if __name__ == "__main__":
  try:
    main()
  finally:
    simulation_app.close()
