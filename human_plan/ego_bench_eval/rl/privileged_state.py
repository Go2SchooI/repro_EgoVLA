from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch


PREFERRED_KEYS = (
  "qpos",
  "qvel",
  "action",
  "left_ee_pose",
  "right_ee_pose",
  "left_target_ee_pose",
  "right_target_ee_pose",
  "left_finger_tip_pos",
  "right_finger_tip_pos",
  "left_hand_contact_force",
  "right_hand_contact_force",
)

EXCLUDED_SUBSTRINGS = (
  "rgb",
  "fixed_d",
  "distance",
  "image",
  "camera",
  "success",
  "reward",
  "done",
)


def _to_numpy_env0(value: Any) -> np.ndarray:
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  arr = np.asarray(value)
  if arr.shape and arr.shape[0] == 1:
    arr = arr[0]
  return arr.astype(np.float32).reshape(-1)


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], str]:
  if isinstance(value, torch.Tensor):
    return tuple(value.shape), str(value.dtype)
  arr = np.asarray(value)
  return tuple(arr.shape), str(arr.dtype)


class PrivilegedStateAdapter:
  def __init__(self, env=None):
    self.env = env
    self.key_schema: list[dict] = []
    self.schema_hash: str | None = None
    self.version_metadata = self._version_metadata()

  def _version_metadata(self) -> dict:
    metadata = {}
    try:
      import omni.isaac.lab as isaac_lab
      metadata["isaac_lab_module"] = str(getattr(isaac_lab, "__file__", "unknown"))
    except Exception:
      pass
    try:
      import omni.isaac.core as isaac_core
      metadata["isaac_core_module"] = str(getattr(isaac_core, "__file__", "unknown"))
    except Exception:
      pass
    return metadata

  def _include_extra_key(self, key: str, value: Any) -> bool:
    lower = key.lower()
    if any(token in lower for token in EXCLUDED_SUBSTRINGS):
      return False
    if key in PREFERRED_KEYS:
      return False
    if not isinstance(value, (torch.Tensor, np.ndarray, float, int, bool)):
      return False
    try:
      arr = _to_numpy_env0(value)
    except Exception:
      return False
    return arr.size > 0 and arr.size <= 256 and np.isfinite(arr).all()

  def build(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
    parts = []
    schema = []
    for key in PREFERRED_KEYS:
      if key not in obs_dict:
        continue
      flat = _to_numpy_env0(obs_dict[key])
      shape, dtype = _shape_dtype(obs_dict[key])
      parts.append(flat)
      schema.append({"key": key, "shape": shape, "dtype": dtype, "flat_dim": int(flat.size)})

    for key in sorted(obs_dict.keys()):
      if not self._include_extra_key(key, obs_dict[key]):
        continue
      flat = _to_numpy_env0(obs_dict[key])
      shape, dtype = _shape_dtype(obs_dict[key])
      parts.append(flat)
      schema.append({"key": key, "shape": shape, "dtype": dtype, "flat_dim": int(flat.size)})

    if not parts:
      raise RuntimeError("PrivilegedStateAdapter found no usable non-image obs fields")

    state = np.concatenate(parts, axis=0).astype(np.float32)
    assert np.isfinite(state).all(), "privileged state contains NaN/Inf"
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    self.key_schema = schema
    self.schema_hash = schema_hash
    meta = {
      "keys": [item["key"] for item in schema],
      "schema": schema,
      "schema_hash": schema_hash,
      "shape": tuple(state.shape),
      "version_metadata": self.version_metadata,
    }
    return state, meta

  def metadata(self) -> dict:
    return {
      "schema": self.key_schema,
      "schema_hash": self.schema_hash,
      "version_metadata": self.version_metadata,
    }
