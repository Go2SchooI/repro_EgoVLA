from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .action_space import ACTION_DIM, ActionNormalizer
from .features import VectorNormalizer, make_fast_fields


FAST_KEYS = (
  "actor_obs",
  "critic_obs",
  "action",
  "ref_action",
  "reward",
  "next_actor_obs",
  "next_critic_obs",
  "done",
)


class RLReplayBuffer:
  def __init__(self, capacity: int, metadata: Optional[dict] = None):
    self.capacity = int(capacity)
    self.metadata = metadata or {}
    self.transitions: list[dict] = []

  def __len__(self) -> int:
    return len(self.transitions)

  def append(self, raw_fields: dict, fast_fields: dict) -> None:
    self._assert_fast(fast_fields)
    transition = {
      "raw": self._copy_np_dict(raw_fields),
      "fast": self._copy_np_dict(fast_fields),
    }
    if len(self.transitions) >= self.capacity:
      self.transitions.pop(0)
    self.transitions.append(transition)

  def _copy_np_dict(self, fields: dict) -> dict:
    out = {}
    for key, value in fields.items():
      if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
      if isinstance(value, np.ndarray):
        out[key] = value.copy()
      else:
        out[key] = np.asarray(value).copy()
    return out

  def _assert_fast(self, fast_fields: dict) -> None:
    missing = [key for key in FAST_KEYS if key not in fast_fields]
    if missing:
      raise KeyError(f"missing fast replay fields: {missing}")
    assert np.asarray(fast_fields["action"]).shape == (ACTION_DIM,)
    assert np.asarray(fast_fields["ref_action"]).shape == (ACTION_DIM,)
    actor_shape = np.asarray(fast_fields["actor_obs"]).shape
    next_actor_shape = np.asarray(fast_fields["next_actor_obs"]).shape
    critic_shape = np.asarray(fast_fields["critic_obs"]).shape
    next_critic_shape = np.asarray(fast_fields["next_critic_obs"]).shape
    assert actor_shape == next_actor_shape, f"actor obs mismatch {actor_shape} vs {next_actor_shape}"
    assert critic_shape == next_critic_shape, f"critic obs mismatch {critic_shape} vs {next_critic_shape}"
    for key in FAST_KEYS:
      assert np.isfinite(np.asarray(fast_fields[key], dtype=np.float32)).all(), f"{key} has NaN/Inf"

  @property
  def actor_obs_dim(self) -> int:
    return int(np.asarray(self.transitions[0]["fast"]["actor_obs"]).shape[0])

  @property
  def critic_obs_dim(self) -> int:
    return int(np.asarray(self.transitions[0]["fast"]["critic_obs"]).shape[0])

  @property
  def action_dim(self) -> int:
    if self.transitions:
      return int(np.asarray(self.transitions[0]["fast"]["action"]).shape[0])
    return int(self.metadata.get("actor_action_dim", ACTION_DIM))

  def sample_batch(self, batch_size: int, device: torch.device | str):
    if len(self.transitions) < batch_size:
      raise RuntimeError(f"not enough replay samples: {len(self.transitions)} < {batch_size}")
    indices = np.random.randint(0, len(self.transitions), size=int(batch_size))
    batch = {}
    for key in FAST_KEYS:
      values = [self.transitions[idx]["fast"][key] for idx in indices]
      stacked = np.stack(values, axis=0).astype(np.float32)
      if key in ("reward", "done"):
        stacked = stacked.reshape(-1, 1)
      batch[key] = torch.from_numpy(stacked).to(device)
    return batch

  def fit_action_normalizer(self, clip: float = 0.999) -> ActionNormalizer:
    refs = np.stack([tr["raw"]["a_ref"] for tr in self.transitions], axis=0).astype(np.float32)
    return ActionNormalizer.fit(refs, clip=clip)

  def rebuild_fast_fields(
    self,
    action_normalizer: ActionNormalizer,
    summary_type: str = "selected_steps",
    selected_steps: tuple[int, ...] = (0, 5, 10, 20, 29),
    feature_hook: str = "traj_decoder_input",
    actor_obs_normalizer: Optional[VectorNormalizer] = None,
    critic_obs_normalizer: Optional[VectorNormalizer] = None,
  ) -> None:
    fast_meta = None
    for transition in self.transitions:
      fast, meta = make_fast_fields(
        transition["raw"],
        action_normalizer,
        summary_type,
        selected_steps,
        feature_hook,
        actor_obs_normalizer,
        critic_obs_normalizer,
      )
      self._assert_fast(fast)
      transition["fast"] = fast
      fast_meta = meta
    self.metadata["action_normalizer"] = action_normalizer.state_dict()
    self.metadata["chunk_summary_type"] = summary_type
    self.metadata["chunk_summary_steps"] = tuple(selected_steps)
    self.metadata["feature_hook"] = feature_hook
    if actor_obs_normalizer is not None:
      self.metadata["actor_obs_normalizer"] = actor_obs_normalizer.state_dict()
    if critic_obs_normalizer is not None:
      self.metadata["critic_obs_normalizer"] = critic_obs_normalizer.state_dict()
    if fast_meta is not None:
      self.metadata["fast_field_shapes"] = fast_meta

  def fit_obs_normalizers(
    self,
    action_normalizer: ActionNormalizer,
    summary_type: str = "selected_steps",
    selected_steps: tuple[int, ...] = (0, 5, 10, 20, 29),
    feature_hook: str = "traj_decoder_input",
    clip: float = 10.0,
  ) -> tuple[VectorNormalizer, VectorNormalizer]:
    actor_obs_values = []
    critic_obs_values = []
    for transition in self.transitions:
      fast, _ = make_fast_fields(
        transition["raw"],
        action_normalizer,
        summary_type,
        selected_steps,
        feature_hook,
      )
      actor_obs_values.append(fast["actor_obs"])
      actor_obs_values.append(fast["next_actor_obs"])
      critic_obs_values.append(fast["critic_obs"])
      critic_obs_values.append(fast["next_critic_obs"])
    actor_obs_normalizer = VectorNormalizer.fit(
      np.stack(actor_obs_values, axis=0),
      clip=clip,
      preserve_tail_dim=ACTION_DIM,
    )
    critic_obs_normalizer = VectorNormalizer.fit(np.stack(critic_obs_values, axis=0), clip=clip)
    return actor_obs_normalizer, critic_obs_normalizer

  def save(self, path: str | Path) -> None:
    path = Path(path)
    if path.parent and str(path.parent) not in ("", "."):
      path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "capacity": self.capacity,
      "metadata": self.metadata,
      "transitions": self.transitions,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

  @classmethod
  def load(cls, path: str | Path) -> "RLReplayBuffer":
    try:
      payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      payload = torch.load(path, map_location="cpu")
    buffer = cls(payload.get("capacity", len(payload["transitions"])), payload.get("metadata", {}))
    buffer.transitions = payload["transitions"]
    for transition in buffer.transitions:
      buffer._assert_fast(transition["fast"])
    return buffer
