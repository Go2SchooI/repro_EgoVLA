from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


ACTION_SLICES = {
  "left_ee": slice(0, 7),
  "right_ee": slice(7, 14),
  "left_hand": slice(14, 26),
  "right_hand": slice(26, 38),
}
ACTION_DIM = ACTION_SLICES["right_hand"].stop


def _to_numpy_1d(value, name: str, dim: int, assert_finite: bool = True) -> np.ndarray:
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  arr = np.asarray(value, dtype=np.float32).reshape(-1)
  assert arr.shape == (dim,), f"{name}.shape must be ({dim},), got {arr.shape}"
  if assert_finite:
    assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"
  return arr


def pack_action(left_ee, right_ee, left_hand, right_hand) -> np.ndarray:
  parts = [
    _to_numpy_1d(left_ee, "left_ee", 7),
    _to_numpy_1d(right_ee, "right_ee", 7),
    _to_numpy_1d(left_hand, "left_hand", 12),
    _to_numpy_1d(right_hand, "right_hand", 12),
  ]
  packed = np.concatenate(parts, axis=0).astype(np.float32)
  assert packed.shape == (ACTION_DIM,), f"packed action shape must be ({ACTION_DIM},), got {packed.shape}"
  return packed


def unpack_action(action: np.ndarray) -> dict[str, np.ndarray]:
  action = _to_numpy_1d(action, "action", ACTION_DIM)
  return {
    key: action[value].copy()
    for key, value in ACTION_SLICES.items()
  }


def build_base_chunk(action_dict: dict) -> np.ndarray:
  left_ee = np.asarray(action_dict["left_ee_pose"], dtype=np.float32)
  right_ee = np.asarray(action_dict["right_ee_pose"], dtype=np.float32)
  left_hand = np.asarray(action_dict["left_qpos_multi_step"], dtype=np.float32)
  right_hand = np.asarray(action_dict["right_qpos_multi_step"], dtype=np.float32)
  assert left_ee.ndim == 2 and left_ee.shape[-1] == 7, f"left_ee_pose shape {left_ee.shape}"
  assert right_ee.ndim == 2 and right_ee.shape[-1] == 7, f"right_ee_pose shape {right_ee.shape}"
  assert left_hand.ndim == 2 and left_hand.shape[-1] == 12, f"left_qpos_multi_step shape {left_hand.shape}"
  assert right_hand.ndim == 2 and right_hand.shape[-1] == 12, f"right_qpos_multi_step shape {right_hand.shape}"
  t = left_ee.shape[0]
  assert right_ee.shape[0] == t and left_hand.shape[0] == t and right_hand.shape[0] == t
  return np.stack(
    [pack_action(left_ee[i], right_ee[i], left_hand[i], right_hand[i]) for i in range(t)],
    axis=0,
  )


@dataclass
class ActionNormalizer:
  mean: np.ndarray
  std: np.ndarray
  eps: float = 1e-6
  clip: float = 0.999
  mode: str = "minmax_clip"

  @classmethod
  def identity(cls, action_dim: int = ACTION_DIM) -> "ActionNormalizer":
    return cls(
      mean=np.zeros(action_dim, dtype=np.float32),
      std=np.ones(action_dim, dtype=np.float32),
      mode="identity",
    )

  @classmethod
  def fit(cls, actions: np.ndarray, eps: float = 1e-6, clip: float = 0.999) -> "ActionNormalizer":
    actions = np.asarray(actions, dtype=np.float32)
    assert actions.ndim == 2 and actions.shape[1] == ACTION_DIM, f"actions shape {actions.shape}"
    low = actions.min(axis=0)
    high = actions.max(axis=0)
    mean = ((high + low) * 0.5).astype(np.float32)
    std = np.maximum((high - low) * 0.5, eps).astype(np.float32)
    return cls(mean=mean, std=std, eps=eps, clip=float(clip), mode="minmax_clip")

  @classmethod
  def from_state(cls, state: Optional[dict]) -> "ActionNormalizer":
    if not state:
      return cls.identity()
    return cls(
      mean=np.asarray(state["mean"], dtype=np.float32),
      std=np.asarray(state["std"], dtype=np.float32),
      eps=float(state.get("eps", 1e-6)),
      clip=float(state.get("clip", 0.999)),
      mode=str(state.get("mode", "legacy_zscore")),
    )

  def state_dict(self) -> dict:
    return {
      "mean": self.mean.astype(np.float32),
      "std": self.std.astype(np.float32),
      "eps": self.eps,
      "clip": self.clip,
      "mode": self.mode,
    }

  def normalize(self, action) -> np.ndarray:
    action = _to_numpy_1d(action, "action", ACTION_DIM)
    out = (action - self.mean) / np.maximum(self.std, self.eps)
    out = np.clip(out, -self.clip, self.clip)
    return out.astype(np.float32)

  def denormalize(self, action_norm) -> np.ndarray:
    action_norm = _to_numpy_1d(action_norm, "action_norm", ACTION_DIM)
    action_norm = np.clip(action_norm, -self.clip, self.clip)
    return (action_norm * np.maximum(self.std, self.eps) + self.mean).astype(np.float32)

  def normalize_torch(self, action: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(self.mean, dtype=action.dtype, device=action.device)
    std = torch.as_tensor(np.maximum(self.std, self.eps), dtype=action.dtype, device=action.device)
    return torch.clamp((action - mean) / std, -self.clip, self.clip)

  def denormalize_torch(self, action_norm: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(self.mean, dtype=action_norm.dtype, device=action_norm.device)
    std = torch.as_tensor(np.maximum(self.std, self.eps), dtype=action_norm.dtype, device=action_norm.device)
    action_norm = torch.clamp(action_norm, -self.clip, self.clip)
    return action_norm * std + mean


def normalize_action(action, normalizer: Optional[ActionNormalizer] = None) -> np.ndarray:
  return (normalizer or ActionNormalizer.identity()).normalize(action)


def denormalize_action(action_norm, normalizer: Optional[ActionNormalizer] = None) -> np.ndarray:
  return (normalizer or ActionNormalizer.identity()).denormalize(action_norm)


def _normalize_quat(quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float32).copy()
  ref_quat = np.asarray(ref_quat, dtype=np.float32)
  if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
    quat = ref_quat.copy()
  norm = np.linalg.norm(quat)
  if norm < 1e-6:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  else:
    quat = quat / norm
  if np.dot(quat, ref_quat) < 0:
    quat = -quat
  return quat.astype(np.float32)


def _hand_limits_for(env, side: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
  if env is None:
    return None
  cfg_name = "left_hand_cfg" if side == "left" else "right_hand_cfg"
  if not hasattr(env, "robot_dof_lower_limits") or not hasattr(env, "robot_dof_upper_limits"):
    return None
  if not hasattr(env, "cfg") or not hasattr(env.cfg, cfg_name):
    return None
  joint_ids = getattr(getattr(env.cfg, cfg_name), "joint_ids", None)
  if joint_ids is None:
    return None
  lower = env.robot_dof_lower_limits[joint_ids]
  upper = env.robot_dof_upper_limits[joint_ids]
  if isinstance(lower, torch.Tensor):
    lower = lower.detach().cpu().numpy()
  if isinstance(upper, torch.Tensor):
    upper = upper.detach().cpu().numpy()
  lower = np.asarray(lower, dtype=np.float32).reshape(-1)
  upper = np.asarray(upper, dtype=np.float32).reshape(-1)
  if lower.shape != (12,) or upper.shape != (12,):
    return None
  return lower, upper


def postprocess_action(action, ref_action, env=None) -> np.ndarray:
  ref = _to_numpy_1d(ref_action, "ref_action", ACTION_DIM)
  out = _to_numpy_1d(action, "action", ACTION_DIM, assert_finite=False).copy()
  bad = ~np.isfinite(out)
  out[bad] = ref[bad]

  left_slice = ACTION_SLICES["left_ee"]
  right_slice = ACTION_SLICES["right_ee"]
  out[left_slice.start + 3:left_slice.start + 7] = _normalize_quat(
    out[left_slice.start + 3:left_slice.start + 7],
    ref[left_slice.start + 3:left_slice.start + 7],
  )
  out[right_slice.start + 3:right_slice.start + 7] = _normalize_quat(
    out[right_slice.start + 3:right_slice.start + 7],
    ref[right_slice.start + 3:right_slice.start + 7],
  )

  left_limits = _hand_limits_for(env, "left")
  if left_limits is not None:
    out[ACTION_SLICES["left_hand"]] = np.clip(out[ACTION_SLICES["left_hand"]], left_limits[0], left_limits[1])
  right_limits = _hand_limits_for(env, "right")
  if right_limits is not None:
    out[ACTION_SLICES["right_hand"]] = np.clip(out[ACTION_SLICES["right_hand"]], right_limits[0], right_limits[1])

  assert np.isfinite(out).all(), "postprocessed action contains NaN/Inf"
  for key in ("left_ee", "right_ee"):
    quat_norm = np.linalg.norm(out[ACTION_SLICES[key]][3:7])
    assert abs(quat_norm - 1.0) < 1e-3, f"{key} quaternion norm is {quat_norm}"
  return out.astype(np.float32)
