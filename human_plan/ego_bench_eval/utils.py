import os
import torch
from omni.isaac.lab.utils.math import subtract_frame_transforms
from human_plan.utils.mano.forward import (
   mano_forward_retarget
)
from llava.data.utils import (
  norm_hand_dof,
  denorm_hand_dof
)

from human_plan.preprocessing.preprocessing import (
  preprocess_vla,
  preprocess_multimodal_vla
)

from human_plan.preprocessing.prompting_format import (
  preprocess_language_instruction,
)

from llava.mm_utils import (
  process_image_ndarray_v2
)

from human_plan.dataset_preprocessing.otv_isaaclab.utils import (
  LANGUAGE_MAPPING,
  LANGUAGE_MAPPING_NEW
)

TASK_MAX_HORIZON = {
  "Humanoid-Push-Box-v0": 400,
  "Humanoid-Open-Drawer-v0": 400,
  "Humanoid-Close-Drawer-v0": 400,
  "Humanoid-Flip-Mug-v0": 400,
  "Humanoid-Pour-Balls-v0": 400,

  "Humanoid-Sort-Cans-v0": 900,
  "Humanoid-Insert-Cans-v0": 500,
  "Humanoid-Unload-Cans-v0": 500,

  "Humanoid-Insert-And-Unload-Cans-v0": 900,
  "Humanoid-Orient-Pour-Balls-v0": 600,

  "Humanoid-Stack-Can-v0": 300,
  "Humanoid-Open-Laptop-v0": 300,
 
  "Humanoid-Stack-Can-Into-Drawer-v0": 400
}

TASK_INIT_EPISODE = {
    "Close-Drawer": [
        ("episode_17.hdf5", 0.008808),
        ("episode_35.hdf5", 0.010483),
        ("episode_39.hdf5", 0.011340),
        ("episode_40.hdf5", 0.011753),
        ("episode_14.hdf5", 0.011799),
        ("episode_3.hdf5", 0.011995),
        ("episode_6.hdf5", 0.012234),
        ("episode_27.hdf5", 0.012507),
        ("episode_41.hdf5", 0.012627),
        ("episode_29.hdf5", 0.012791),
    ],
    "Flip-Mug": [
        ("episode_49.hdf5", 0.002252),
        ("episode_47.hdf5", 0.002291),
        ("episode_15.hdf5", 0.002464),
        ("episode_26.hdf5", 0.002471),
        ("episode_22.hdf5", 0.002494),
        ("episode_48.hdf5", 0.002598),
        ("episode_20.hdf5", 0.002886),
        ("episode_17.hdf5", 0.002915),
        ("episode_24.hdf5", 0.003137),
        ("episode_40.hdf5", 0.003161),
    ],
    "Insert-And-Unload-Cans": [
        ("episode_98.hdf5", 0.002737),
        ("episode_62.hdf5", 0.003789),
        ("episode_64.hdf5", 0.004229),
        ("episode_93.hdf5", 0.004603),
        ("episode_74.hdf5", 0.005061),
        ("episode_61.hdf5", 0.005299),
        ("episode_65.hdf5", 0.005371),
        ("episode_57.hdf5", 0.005518),
        ("episode_66.hdf5", 0.005821),
        ("episode_81.hdf5", 0.005871),
    ],
    "Insert-Cans": [
        ("episode_56.hdf5", 0.006791),
        ("episode_62.hdf5", 0.007331),
        ("episode_68.hdf5", 0.007532),
        ("episode_51.hdf5", 0.007645),
        ("episode_57.hdf5", 0.007861),
        ("episode_60.hdf5", 0.007894),
        ("episode_67.hdf5", 0.007988),
        ("episode_64.hdf5", 0.008021),
        ("episode_66.hdf5", 0.008172),
        ("episode_24.hdf5", 0.008201),
    ],
    "Open-Drawer": [
        ("episode_35.hdf5", 0.008118),
        ("episode_0.hdf5", 0.010806),
        ("episode_42.hdf5", 0.010956),
        ("episode_11.hdf5", 0.011135),
        ("episode_14.hdf5", 0.011418),
        ("episode_32.hdf5", 0.011524),
        ("episode_4.hdf5", 0.012320),
        ("episode_2.hdf5", 0.012502),
        ("episode_37.hdf5", 0.012656),
        ("episode_23.hdf5", 0.012837),
    ],
    "Open-Laptop": [
        ("episode_8.hdf5", 0.008424),
        ("episode_35.hdf5", 0.008433),
        ("episode_25.hdf5", 0.009747),
        ("episode_90.hdf5", 0.009884),
        ("episode_5.hdf5", 0.010008),
        ("episode_72.hdf5", 0.010378),
        ("episode_45.hdf5", 0.010485),
        ("episode_6.hdf5", 0.010558),
        ("episode_41.hdf5", 0.010697),
        ("episode_9.hdf5", 0.010750),
    ],
    "Pour-Balls": [
        ("episode_55.hdf5", 0.005932),
        ("episode_52.hdf5", 0.006337),
        ("episode_63.hdf5", 0.006806),
        ("episode_66.hdf5", 0.006951),
        ("episode_65.hdf5", 0.007088),
        ("episode_54.hdf5", 0.007339),
        ("episode_61.hdf5", 0.007532),
        ("episode_74.hdf5", 0.007791),
        ("episode_60.hdf5", 0.008137),
        ("episode_64.hdf5", 0.008280),
    ],
    "Push-Box": [
        ("episode_50.hdf5", 0.010576),
        ("episode_51.hdf5", 0.011270),
        ("episode_52.hdf5", 0.011941),
        ("episode_98.hdf5", 0.012541),
        ("episode_66.hdf5", 0.012898),
        ("episode_70.hdf5", 0.012963),
        ("episode_53.hdf5", 0.013595),
        ("episode_92.hdf5", 0.013633),
        ("episode_97.hdf5", 0.013849),
        ("episode_6.hdf5", 0.013854),
    ],
    "Sort-Cans": [
        ("episode_1.hdf5", 0.011536),
        ("episode_58.hdf5", 0.012234),
        ("episode_28.hdf5", 0.012709),
        ("episode_16.hdf5", 0.012919),
        ("episode_30.hdf5", 0.012944),
        ("episode_21.hdf5", 0.013026),
        ("episode_29.hdf5", 0.013076),
        ("episode_47.hdf5", 0.013148),
        ("episode_27.hdf5", 0.013332),
        ("episode_26.hdf5", 0.013394),
    ],
    "Stack-Can": [
        ("episode_55.hdf5", 0.006748),
        ("episode_32.hdf5", 0.007932),
        ("episode_43.hdf5", 0.007939),
        ("episode_44.hdf5", 0.007952),
        ("episode_91.hdf5", 0.008753),
        ("episode_20.hdf5", 0.008928),
        ("episode_47.hdf5", 0.009045),
        ("episode_22.hdf5", 0.009379),
        ("episode_12.hdf5", 0.009754),
        ("episode_23.hdf5", 0.009801),
    ],
    "Stack-Can-Into-Drawer": [
        ("episode_28.hdf5", 0.037548),
        ("episode_39.hdf5", 0.042628),
        ("episode_38.hdf5", 0.051440),
        ("episode_29.hdf5", 0.052506),
        ("episode_43.hdf5", 0.056320),
        ("episode_30.hdf5", 0.061978),
        ("episode_35.hdf5", 0.062655),
        ("episode_15.hdf5", 0.064243),
        ("episode_36.hdf5", 0.067350),
        ("episode_13.hdf5", 0.068193),
    ],
    "Unload-Cans": [
        ("episode_47.hdf5", 0.003218),
        ("episode_37.hdf5", 0.004854),
        ("episode_24.hdf5", 0.005575),
        ("episode_30.hdf5", 0.006169),
        ("episode_29.hdf5", 0.006214),
        ("episode_39.hdf5", 0.006806),
        ("episode_23.hdf5", 0.006823),
        ("episode_2.hdf5", 0.006844),
        ("episode_32.hdf5", 0.006960),
        ("episode_33.hdf5", 0.007112),
    ],
}

def get_language_instruction(task):
  task_name = task[9:-3]
  return LANGUAGE_MAPPING[task_name]

from human_plan.dataset_preprocessing.utils.mano_utils import (
  obtain_mano_pose_otv_inspire_single_step
)

from human_plan.utils.visualization import (
   project_points,
)

from human_plan.dataset_preprocessing.otv_isaaclab.utils import (
  to_cam_frame,
  to_pose,
  pose_to_cam_frame,
  robot_action_to_mano_action
)

def process_proprio_input(
    left_finger_tip,
    right_finger_tip,
    left_ee_pose,
    right_ee_pose,
    current_qpos,
    cam_intrinsics,
    input_hand_dof = True
):
  
  left_ee_pose_cam_frame = pose_to_cam_frame(
    to_pose(left_ee_pose)
  )
  right_ee_pose_cam_frame = pose_to_cam_frame(
    to_pose(right_ee_pose)
  )
  # Finger tip pos
  left_finger_tip_pos_cam_frame = to_cam_frame(
    left_finger_tip
  )
  right_finger_tip_pos_cam_frame = to_cam_frame(
    right_finger_tip
  )

  mano_kps3d_right = obtain_mano_pose_otv_inspire_single_step(
    right_ee_pose_cam_frame, is_right=True
  )
  mano_kps3d_left = obtain_mano_pose_otv_inspire_single_step(
    left_ee_pose_cam_frame, is_right=True
  )

  right_mano_ee_2d = project_points(
    right_ee_pose_cam_frame[:, :3, -1], cam_intrinsics
  )

  left_mano_ee_2d = project_points(
    left_ee_pose_cam_frame[:, :3, -1], cam_intrinsics
  )

  mano_dof = infer_retarget_to_mano(
      hand_mano_retarget_net, 
      current_qpos
  )
  mano_dof_left = mano_dof[:15]
  mano_dof_left = norm_hand_dof(torch.Tensor(mano_dof_left))
  mano_dof_left = norm_hand_dof(mano_dof_left)

  mano_dof_right = mano_dof[15:]
  mano_dof_right = norm_hand_dof(torch.Tensor(mano_dof_right))
  mano_dof_right = norm_hand_dof(mano_dof_right)

  current_proprio = {
    "left_hand_pose": {
      # "mano_kps3d": mano_kps3d_left["mano_kp_predicted"],
      # "mano_parameters": mano_kps3d_left["optimized_mano_parameters"],
      "mano_parameters": mano_dof_left,
      "mano_rot": mano_kps3d_left["optimized_mano_rot"],
      "mano_trans": mano_kps3d_left["optimized_mano_trans"],
      "mano_ee_2d": left_mano_ee_2d,
      "finger_tip": left_finger_tip_pos_cam_frame
    }, 
    "right_hand_pose": {
      # "mano_kps3d": mano_kps3d_right["mano_kp_predicted"],
      # "mano_parameters": mano_kps3d_right["optimized_mano_parameters"],
      "mano_parameters": mano_dof_right,
      "mano_rot": mano_kps3d_right["optimized_mano_rot"],
      "mano_trans": mano_kps3d_right["optimized_mano_trans"],
      "mano_ee_2d": right_mano_ee_2d,
      "finger_tip": right_finger_tip_pos_cam_frame
    }
  }

  ee_2d_inputs = torch.concat([
      torch.tensor(current_proprio["left_hand_pose"]["mano_ee_2d"], device="cuda").float(),
      torch.tensor(current_proprio["right_hand_pose"]["mano_ee_2d"], device="cuda").float()
  ], dim=-1)

  ee_3d_inputs = torch.concat([
      torch.tensor(current_proprio["left_hand_pose"]["mano_trans"], device="cuda").float(),
      torch.tensor(current_proprio["right_hand_pose"]["mano_trans"], device="cuda").float()
  ], dim=-1)

  ee_rot_inputs = torch.concat([
      torch.tensor(current_proprio["left_hand_pose"]["mano_rot"], device="cuda").float(),
      torch.tensor(current_proprio["right_hand_pose"]["mano_rot"], device="cuda").float()
  ], dim=-1)

  hand_dof_inputs = torch.concat([
      torch.tensor(current_proprio["left_hand_pose"]["mano_parameters"], device="cuda").float(),
      torch.tensor(current_proprio["right_hand_pose"]["mano_parameters"], device="cuda").float()
  ], dim=-1)

  finger_tip_inputs = torch.concat([
      torch.tensor(current_proprio["left_hand_pose"]["finger_tip"], device="cuda").float(),
      torch.tensor(current_proprio["right_hand_pose"]["finger_tip"], device="cuda").float()
  ], dim=-1)


  if input_hand_dof:
    proprio_input = torch.cat([
          ee_3d_inputs.reshape(-1, 6),
          ee_rot_inputs.reshape(-1, 6),
          hand_dof_inputs.reshape(-1, 30)
    ], dim=-1)
  else:
    proprio_input = torch.cat([
          ee_2d_inputs.reshape(-1, 4),
          ee_3d_inputs.reshape(-1, 6),
          ee_rot_inputs.reshape(-1, 6)
    ], dim=-1)

  raw_proprio_inputs = current_proprio
  raw_proprio_inputs.update({
    "proprio_input_2d": ee_2d_inputs.reshape(-1, 4),
    "proprio_input_3d": ee_3d_inputs.reshape(-1, 6),
    "proprio_input_rot": ee_rot_inputs.reshape(-1, 6),
    "proprio_input_handdof": hand_dof_inputs.reshape(-1, 30),
    "proprio_input_hand_finger_tip": finger_tip_inputs.reshape(-1, 5 * 3 * 2)
  })
  return proprio_input, raw_proprio_inputs

def ik_step(
    env,
    left_ik_controller,
    right_ik_controller,

    left_ik_commands_world,
    right_ik_commands_world,
    
    left_ik_commands_robot,
    right_ik_commands_robot,

    left_ee_goal, right_ee_goal,
    left_hand_dof, right_hand_dof,
    
    action
):
  left_jacobin_idx = env.left_ee_idx-1
  right_jacobin_idx = env.right_ee_idx-1

  robot_pose_w = env.robot.data.root_state_w[:, 0:7]
  left_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, left_jacobin_idx, :, env.cfg.left_arm_cfg.joint_ids]
  left_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.left_arm_cfg.body_ids[0], 0:7]
  left_joint_pos = env.robot.data.joint_pos[:, env.cfg.left_arm_cfg.joint_ids]
  right_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, right_jacobin_idx, :, env.cfg.right_arm_cfg.joint_ids]
  right_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.right_arm_cfg.body_ids[0], 0:7]
  right_joint_pos = env.robot.data.joint_pos[:, env.cfg.right_arm_cfg.joint_ids]
  # prepare IK 
  left_ee_curr_pose_robot, left_ee_curr_quat_robot = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ee_curr_pose_world[:, 0:3], left_ee_curr_pose_world[:, 3:7]
  )
  right_ee_curr_pos_robot, right_ee_curr_quat_robot = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ee_curr_pose_world[:, 0:3], right_ee_curr_pose_world[:, 3:7]
  )
  left_ik_commands_world[:, 0:7] = torch.FloatTensor(left_ee_goal[0:7]).to(left_ik_commands_world.device)
  left_ik_commands_robot[:, 0:3], left_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ik_commands_world[:, 0:3], left_ik_commands_world[:, 3:7]
  )
  right_ik_commands_world[:, 0:7] = torch.FloatTensor(right_ee_goal[0:7]).to(right_ik_commands_world.device)
  right_ik_commands_robot[:, 0:3], right_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
      robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ik_commands_world[:, 0:3], right_ik_commands_world[:, 3:7]
  )
  left_ik_controller.set_command(left_ik_commands_robot, left_ee_curr_pose_robot, left_ee_curr_quat_robot)
  right_ik_controller.set_command(right_ik_commands_robot, right_ee_curr_pos_robot, right_ee_curr_quat_robot)
  # compute the joint commands
  left_joint_pos_des = left_ik_controller.compute(left_ee_curr_pose_robot, left_ee_curr_quat_robot, left_arm_jacobian, left_joint_pos)
  right_joint_pos_des = right_ik_controller.compute(right_ee_curr_pos_robot, right_ee_curr_quat_robot, right_arm_jacobian, right_joint_pos)

  action[:, :] = 0
  action[:, env.cfg.left_arm_cfg.joint_ids] = left_joint_pos_des
  action[:, env.cfg.right_arm_cfg.joint_ids] = right_joint_pos_des

  action[:, env.cfg.left_hand_cfg.joint_ids] =  torch.FloatTensor(left_hand_dof).to(action.device)# set hand joint commands
  action[:, env.cfg.right_hand_cfg.joint_ids] =  torch.FloatTensor(right_hand_dof).to(action.device)# set hand joint commands



from human_plan.vila_eval.utils.eval_func import (
  eval_single_sample,
  to_ndarray
)

import numpy as np
from human_plan.dataset_preprocessing.otv_isaaclab.utils import (
  main_cam_transformation,
  CAM_AXIS_TRANSFORM
)

def cam_frame_poses_to_world_frame(poses):
  poses = main_cam_transformation @ np.linalg.inv(CAM_AXIS_TRANSFORM) @ poses.reshape(-1, 4, 4)
  return poses


from human_plan.utils.mano.constants import (
  RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB,
  LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB,
  LEFT_PELVIS,
  RIGHT_PELVIS
)

from scipy.spatial.transform import Rotation as R
from human_plan.dataset_preprocessing.otv_isaaclab.utils import (
  main_cam_transformation
)

def ee_pose_from_mano_pose(
  mano_rot, mano_trans, is_right  
):

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS

  mano_rot_R = R.from_rotvec(mano_rot)
  mano_rot_mat = mano_rot_R.as_matrix()
  ee_rot = mano_rot_mat @ np.linalg.inv(retarget_axis_transformation.cpu().numpy())
  global_trans = mano_trans + pelvis.cpu().numpy()


  transformation = np.zeros((mano_rot.shape[0], 4, 4))
  transformation[:, -1, -1] = 1
  transformation[..., :3, :3] = ee_rot
  transformation[..., :3, -1] = global_trans

  world_frame_transformation = cam_frame_poses_to_world_frame(transformation)

  ee_rot_quat = R.from_matrix(world_frame_transformation[:, :3, :3])
  # xyz W
  ee_rot_quat = ee_rot_quat.as_quat()
  ee_rot_quat_wxyz = np.concatenate([
    ee_rot_quat[..., 3:4], ee_rot_quat[..., :3],
  ], axis=-1)

  world_frame_trans = world_frame_transformation[:, :3, -1]

  ee_pose_isaaclab = np.concatenate([
    world_frame_trans, ee_rot_quat_wxyz
  ], axis=-1)

  return ee_pose_isaaclab


def ee_pose_from_mano_pose_rotmat(
  mano_rot_mat, mano_trans, is_right  
):

  retarget_axis_transformation = RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB if is_right else \
    LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB

  pelvis = RIGHT_PELVIS if is_right else LEFT_PELVIS

  ee_rot = mano_rot_mat @ np.linalg.inv(retarget_axis_transformation.cpu().numpy())
  global_trans = mano_trans + pelvis.cpu().numpy()

  transformation = np.zeros((mano_rot_mat.shape[0], 4, 4))
  transformation[:, -1, -1] = 1
  transformation[..., :3, :3] = ee_rot
  transformation[..., :3, -1] = global_trans

  world_frame_transformation = cam_frame_poses_to_world_frame(transformation)

  ee_rot_quat = R.from_matrix(world_frame_transformation[:, :3, :3])
  # xyz W
  ee_rot_quat = ee_rot_quat.as_quat()
  ee_rot_quat_wxyz = np.concatenate([
    ee_rot_quat[..., 3:4], ee_rot_quat[..., :3],
  ], axis=-1)

  world_frame_trans = world_frame_transformation[:, :3, -1]

  ee_pose_isaaclab = np.concatenate([
    world_frame_trans, ee_rot_quat_wxyz
  ], axis=-1)

  return ee_pose_isaaclab

reverse_channel_order = False

ISAAC_LAB_FRAME_SKIP = 2

def process_input(
  rgb_obs_hist,
  proprio_input,
  raw_language_instruction,
  data_args,
  model_args, tokenizer,
):
  # Sim data is 30 HZ
  frame_count_scaler_up = 1
  frame_count_scaler = 1
  valid_hist_len = len(rgb_obs_hist)
  rgb_obs = []
  rgb_obs_his = []
  for idx in range(data_args.add_his_obs_step):
    rgb_image_idx = max(
      0,
      (valid_hist_len) - (idx + 1) * ISAAC_LAB_FRAME_SKIP * data_args.add_his_img_skip * \
        frame_count_scaler_up // frame_count_scaler
    )
    # print("Loading:", rgb_image_idx)
    rgb_obs.append(process_image_ndarray_v2(
        rgb_obs_hist[rgb_image_idx],
        data_args,
        reverse_channel_order=reverse_channel_order
      )
    )
    rgb_obs_his.append(
      rgb_obs_hist[rgb_image_idx]
    )
  valid_his_len = data_args.add_his_obs_step

  rgb_obs.append(process_image_ndarray_v2(
      rgb_obs_hist[-1],
      data_args,
      reverse_channel_order=reverse_channel_order
  ))
  rgb_obs_his.append(
    rgb_obs_hist[-1]
  )
  image = torch.stack(rgb_obs, dim=0)

  language_instruction = preprocess_language_instruction(
    raw_language_instruction, valid_his_len, data_args
  )

  language_instruction = preprocess_multimodal_vla(
    language_instruction,
    data_args
  )

  hand_label_place_holder = torch.zeros((
    data_args.predict_future_step, 4 + 6 + 6 + 30
  ))
  mask_place_holder = torch.ones((
    data_args.predict_future_step, 4 + 6 + 6 + 30
  )).bool()

  data_dict = preprocess_vla(
    language_instruction,
    proprio_input,
    hand_label_place_holder,
    mask_place_holder,
    data_args.action_tokenizer,
    tokenizer,
    mask_input=data_args.mask_input,
    mask_ignore=data_args.mask_ignore,
    raw_action_label=data_args.raw_action_label,
    traj_action_output_dim=data_args.traj_action_output_dim,
    input_placeholder_diff_index=data_args.input_placeholder_diff_index,
    sep_query_token=data_args.sep_query_token,
    language_response=None,
    include_response=data_args.include_response,
    include_repeat_instruction=data_args.include_repeat_instruction,
    raw_language_label=raw_language_instruction
  )

  data_dict["image"] = image

  data_dict["raw_rgb_obs_his"] = rgb_obs_his
  data_dict["raw_width"] = 1280
  data_dict["raw_height"] = 720
  data_dict["raw_image_obs"] = rgb_obs_hist[-1]

  data_dict["ee_movement_mask"] = torch.ones(
    1, 2
  )

  return data_dict


def get_smooth_action_weight(hist_len, smooth_weight):
  action_weight = np.arange(hist_len)
  action_weight = np.exp(-smooth_weight * action_weight)
  action_weight = action_weight / np.sum(action_weight)
  action_weight = action_weight.reshape(-1, 1)
  return action_weight

def repeat_action(
    action, repeat
):
  action = np.repeat(
    action, repeat, axis=0
  )
  return action


def smooth_action(hist_len, smooth_weight, action_deque):
  final_action_list = []
  valid_hist_len = min(len(action_deque), hist_len)

  action_weight = get_smooth_action_weight(valid_hist_len, smooth_weight)

  for i in range(valid_hist_len):
    hist_item_id = valid_hist_len - i - 1
    hist_data = action_deque[hist_item_id][i]
    final_action_list.append(hist_data)

  final_action_list = np.stack(final_action_list)
  final_action = np.sum(final_action_list * action_weight, axis=0)
  return final_action


from human_plan.utils.nn_retarget import (
    HandActuationNet,
    infer,
)

from human_plan.utils.nn_retarget_tomano import (
    # HandActuationNet,
    infer_retarget_to_mano,
)

from human_plan.dataset_preprocessing.utils.mano_utils import (
  mano_to_inspire_mapping
)

EVAL_DEVICE = torch.device(os.environ.get("EGO_VLA_EVAL_DEVICE", "cuda"))

hand_actuation_net = HandActuationNet(
    input_dim=30, output_dim=12 * 2
) 
hand_actuation_net.load_state_dict(
    torch.load("hand_actuation_net.pth", map_location=EVAL_DEVICE)
)
hand_actuation_net.to(EVAL_DEVICE)


hand_mano_retarget_net = HandActuationNet(
    input_dim=12 * 2, output_dim=15 * 2
) 
hand_mano_retarget_net.load_state_dict(
    torch.load("hand_mano_retarget_net.pth", map_location=EVAL_DEVICE)
)
hand_mano_retarget_net.to(EVAL_DEVICE)

from llava.model.language_model.rotation_convert import rot6d_to_rotmat, batch_axis2matrix, batch_matrix2axis

def ik_eval_single_step(
  raw_data_dict,
  model, tokenizer,
  return_rl_features=False,
):
  results = eval_single_sample(
    raw_data_dict,
    tokenizer, model,
    image_width=raw_data_dict["raw_width"],
    image_height=raw_data_dict["raw_height"],
    return_rl_features=return_rl_features,
  )

  if return_rl_features:
    pred, result_img, action_labels, action_masks, loss, rl_features = results
  else:
    pred, result_img, action_labels, action_masks, loss = results
    rl_features = None

  assert pred.ndim == 2, f"pred.ndim must be 2, got {pred.shape}"
  assert pred.shape[-1] == 48, f"current EgoVLA sim eval expects pred shape (T,48), got {pred.shape}"

  # N, 2, 2
  # pred_2d = pred[:, :4]
  # N, 2, 3
  pred_3d = pred[:, :6].reshape(-1, 2, 3)
  # N, 2, 15
  pred_hand = pred[:, 6:36].reshape(-1, 2, 15)

  # Use Rot 6d
  # N, 2, 6
  pred_rot = pred[:, 36:].reshape(-1, 2, 12)
  pred_rotmat = rot6d_to_rotmat(
    torch.tensor(pred_rot)
  ).view(-1, 2, 3, 3)
  
  use_rot_6d = True

  left_denormed_dof = denorm_hand_dof(
    torch.tensor(pred_hand[:, 0, :])#.to("cuda").float()
  ).to(EVAL_DEVICE).float()
  left_denormed_dof[..., 6:] = 0

  right_denormed_dof = denorm_hand_dof(
    torch.tensor(pred_hand[:, 1, :])#.to("cuda").float()
  ).to(EVAL_DEVICE).float()
  right_denormed_dof[..., 6:] = 0

  left_hand3d_kps_forretarget = mano_forward_retarget(
    left_denormed_dof,
    is_right=False
  ).detach()
  left_hand3d_kps_forretarget = left_hand3d_kps_forretarget[:, mano_to_inspire_mapping][:, 1:]

  right_hand3d_kps_forretarget = mano_forward_retarget(
    right_denormed_dof,
    is_right=True
  ).detach()
  right_hand3d_kps_forretarget = right_hand3d_kps_forretarget[:, mano_to_inspire_mapping][:, 1:]

  hand_dof = infer(
      hand_actuation_net, 
      left_hand3d_kps_forretarget.reshape(-1, 15),
      right_hand3d_kps_forretarget.reshape(-1, 15)
  )
  left_qpos_multi_step = hand_dof[:, :12]
  right_qpos_multi_step = hand_dof[:, 12:]
  left_qpos = left_qpos_multi_step[0]
  right_qpos = right_qpos_multi_step[0]

  left_ee_pose = ee_pose_from_mano_pose_rotmat(
    pred_rotmat[:, 0, :], pred_3d[:, 0, :], is_right=False
  )

  right_ee_pose = ee_pose_from_mano_pose_rotmat(
    pred_rotmat[:, 1, :], pred_3d[:, 1, :], is_right=True
  )

  output_dict = dict(
    left_ee_pose = left_ee_pose,
    right_ee_pose = right_ee_pose,
    left_qpos = left_qpos,
    right_qpos = right_qpos,
    left_qpos_multi_step = left_qpos_multi_step,
    right_qpos_multi_step = right_qpos_multi_step,
    left_ee_trans_cam = pred_3d[:, 0, :],
    right_ee_trans_cam = pred_3d[:, 1, :],
    pred_3d = pred_3d,
    pred = pred,
  )
  if return_rl_features:
    output_dict["rl_features"] = rl_features or {}
  return output_dict
