import subprocess
from pathlib import Path

# Define an array of strings (arguments)
task_names = [
  "Humanoid-Push-Box-v0",
  "Humanoid-Open-Drawer-v0",
  "Humanoid-Close-Drawer-v0",
  "Humanoid-Pour-Balls-v0",
  "Humanoid-Flip-Mug-v0",
  "Humanoid-Open-Laptop-v0",
  "Humanoid-Stack-Can-v0",
  "Humanoid-Unload-Cans-v0",
  "Humanoid-Insert-Cans-v0",
  "Humanoid-Stack-Can-Into-Drawer-v0",
  "Humanoid-Sort-Cans-v0",
  "Humanoid-Insert-And-Unload-Cans-v0",
]

room_indexes = ["1", "2", "3", "4", "5"]
table_indexes = ["1", "2", "3", "4", "5"]

combination = []

# The following combination are seen during training
# So remove them from the evaluation to see the generalization ability
for room_idx in room_indexes:
  for table_idx in table_indexes:
    if (room_idx == "1" or room_idx == "2" or room_idx == "3") and table_idx == "1":
        continue
    combination.append((room_idx, table_idx))

print(combination)

num_episodes = 3
num_trials = 1

smooth_weight = "0.2"
hand_smooth_weight = "0.8"

# Path to your Bash script
repo_root = Path(__file__).resolve().parents[2]
script_path = repo_root / "human_plan" / "ego_bench_eval" / "fullpretrain_p30_h5_transv2.sh"

if not script_path.exists():
  raise FileNotFoundError(f"Missing eval script: {script_path}")

project_trajs = "0"
save_frames = "0"

use_per_step_instruction = "0"

video_saving_path = f"playground_clean_eval/diverse_eval_aug_transv2_difftableroom_12345_hoi4dhot3dholotaco_right_retry/"
additional_label = f"diverse_eval_aug_transv2_difftableroom_12345_hoi4dhot3dholotaco_right_retry"
result_saving_path = f"playground_clean_eval/diverse_eval_aug_transv2_difftableroom_12345_hoi4dhot3dholotaco_right_smooth_retry_{smooth_weight}_{hand_smooth_weight}.txt"

# Iterate through the array and run the script with each argument
from tqdm import tqdm
for task_name in tqdm(task_names, desc="Task Name"):
  for room_idx, table_idx in tqdm(combination, desc="Room and Table"):
      print("Room Idx:", room_idx)
      print("Table Idx:", table_idx)
      try:
          # Run the Bash script with the current argument
          print(
            "bash", script_path, task_name, room_idx, table_idx, smooth_weight, str(num_episodes), str(num_trials), result_saving_path, save_frames, project_trajs, hand_smooth_weight, video_saving_path, additional_label, use_per_step_instruction
          )
          result = subprocess.run(
              ["bash", str(script_path), task_name, room_idx, table_idx, smooth_weight, str(num_episodes), str(num_trials), result_saving_path, save_frames, project_trajs, hand_smooth_weight, video_saving_path, additional_label, use_per_step_instruction ],  # Command to run the Bash script
              capture_output=True,          # Capture the output
              text=True,                    # Return output as string
              check=True,                   # Raise an error if the command fails
              cwd=repo_root
          )
      except subprocess.CalledProcessError as e:
          print(f"Error for '{task_name} {room_idx}':\n{e.stderr}")
