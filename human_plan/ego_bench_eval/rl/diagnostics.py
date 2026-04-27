from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .action_space import ACTION_DIM


ACTION_GROUP_SLICES = {
  "left_ee_pos": slice(0, 3),
  "left_ee_rot": slice(3, 7),
  "right_ee_pos": slice(7, 10),
  "right_ee_rot": slice(10, 14),
  "left_hand": slice(14, 26),
  "right_hand": slice(26, 38),
}


def _as_action(value, name: str) -> np.ndarray:
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  arr = np.asarray(value, dtype=np.float32).reshape(-1)
  assert arr.shape == (ACTION_DIM,), f"{name}.shape must be ({ACTION_DIM},), got {arr.shape}"
  assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"
  return arr


def _json_list(value: np.ndarray) -> str:
  return json.dumps(np.asarray(value, dtype=np.float32).round(8).tolist(), separators=(",", ":"))


def action_diff_rows(
  *,
  step: int,
  a_ref,
  a_exec,
  a_actor=None,
  method: str,
  episode: str,
  trial: int,
  room_idx: int,
  table_idx: int,
  checkpoint: Optional[str] = None,
  residual_scale: Optional[float] = None,
  noise_scale: Optional[float] = None,
) -> list[dict]:
  ref = _as_action(a_ref, "a_ref")
  exec_action = _as_action(a_exec, "a_exec")
  actor_action = _as_action(a_actor, "a_actor") if a_actor is not None else None
  diff = exec_action - ref
  actor_diff = actor_action - ref if actor_action is not None else None

  rows = []
  for group_name, group_slice in ACTION_GROUP_SLICES.items():
    group_diff = diff[group_slice]
    row = {
      "step": int(step),
      "group_name": group_name,
      "method": method,
      "episode": episode,
      "trial": int(trial),
      "room_idx": int(room_idx),
      "table_idx": int(table_idx),
      "checkpoint": checkpoint or "",
      "lambda": "" if residual_scale is None else float(residual_scale),
      "noise_scale": "" if noise_scale is None else float(noise_scale),
      "mean_abs": float(np.mean(np.abs(group_diff))),
      "l2": float(np.linalg.norm(group_diff)),
      "max_abs": float(np.max(np.abs(group_diff))),
      "a_ref_slice": _json_list(ref[group_slice]),
      "a_exec_slice": _json_list(exec_action[group_slice]),
      "a_actor_slice": _json_list(actor_action[group_slice]) if actor_action is not None else "",
    }
    if actor_diff is not None:
      actor_group_diff = actor_diff[group_slice]
      row["actor_minus_ref_mean_abs"] = float(np.mean(np.abs(actor_group_diff)))
      row["actor_minus_ref_l2"] = float(np.linalg.norm(actor_group_diff))
      row["actor_minus_ref_max_abs"] = float(np.max(np.abs(actor_group_diff)))
    else:
      row["actor_minus_ref_mean_abs"] = ""
      row["actor_minus_ref_l2"] = ""
      row["actor_minus_ref_max_abs"] = ""
    rows.append(row)
  return rows


def summarize_action_diff_rows(
  rows: list[dict],
  *,
  step_start: int = -1,
  step_end: int = -1,
) -> dict:
  filtered = []
  for row in rows:
    step = int(row["step"])
    if step_start >= 0 and step < step_start:
      continue
    if step_end >= 0 and step > step_end:
      continue
    filtered.append(row)
  rows = filtered
  if not rows:
    return {}
  summary = {}
  for group_name in ACTION_GROUP_SLICES:
    group_rows = [row for row in rows if row["group_name"] == group_name]
    if not group_rows:
      continue
    summary[group_name] = {
      "mean_abs": float(np.mean([float(row["mean_abs"]) for row in group_rows])),
      "l2": float(np.mean([float(row["l2"]) for row in group_rows])),
      "max_abs": float(np.max([float(row["max_abs"]) for row in group_rows])),
    }
  all_mean_abs = [float(row["mean_abs"]) for row in rows]
  all_max_abs = [float(row["max_abs"]) for row in rows]
  summary["all_groups"] = {
    "mean_abs": float(np.mean(all_mean_abs)),
    "max_abs": float(np.max(all_max_abs)),
    "num_rows": int(len(rows)),
  }
  return summary


def identity_error_summary(a_ref, a_exec) -> dict:
  ref = _as_action(a_ref, "a_ref")
  exec_action = _as_action(a_exec, "a_exec")
  diff = exec_action - ref
  per_group = {}
  for group_name, group_slice in ACTION_GROUP_SLICES.items():
    group_diff = diff[group_slice]
    per_group[group_name] = {
      "mean_abs": float(np.mean(np.abs(group_diff))),
      "max_abs": float(np.max(np.abs(group_diff))),
    }
  return {
    "identity_max_abs_error": float(np.max(np.abs(diff))),
    "identity_mean_abs_error": float(np.mean(np.abs(diff))),
    "per_group_identity_error": per_group,
  }


def write_action_diff_logs(path: str | Path, rows: list[dict], metadata: Optional[dict] = None) -> None:
  if not rows:
    return
  path = Path(path)
  if str(path.parent) not in ("", "."):
    path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with path.open("w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  pt_path = path.with_suffix(".pt")
  torch.save(
    {
      "rows": rows,
      "metadata": metadata or {},
    },
    pt_path,
  )
