from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .action_space import ACTION_DIM, ActionNormalizer


@dataclass
class VectorNormalizer:
  mean: np.ndarray
  std: np.ndarray
  eps: float = 1e-6
  clip: float = 10.0
  preserve_tail_dim: int = 0

  @classmethod
  def fit(
    cls,
    values: np.ndarray,
    eps: float = 1e-6,
    clip: float = 10.0,
    preserve_tail_dim: int = 0,
  ) -> "VectorNormalizer":
    values = np.asarray(values, dtype=np.float32)
    assert values.ndim == 2, f"normalizer values must be 2D, got {values.shape}"
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    preserve_tail_dim = int(preserve_tail_dim)
    if preserve_tail_dim > 0:
      assert preserve_tail_dim <= mean.shape[0], (
        f"preserve_tail_dim={preserve_tail_dim} exceeds vector dim {mean.shape[0]}"
      )
      mean[-preserve_tail_dim:] = 0.0
      std[-preserve_tail_dim:] = 1.0
    return cls(mean=mean, std=std, eps=eps, clip=float(clip), preserve_tail_dim=preserve_tail_dim)

  @classmethod
  def from_state(cls, state: Optional[dict]) -> Optional["VectorNormalizer"]:
    if not state:
      return None
    return cls(
      mean=np.asarray(state["mean"], dtype=np.float32),
      std=np.asarray(state["std"], dtype=np.float32),
      eps=float(state.get("eps", 1e-6)),
      clip=float(state.get("clip", 10.0)),
      preserve_tail_dim=int(state.get("preserve_tail_dim", 0)),
    )

  def state_dict(self) -> dict:
    return {
      "mean": self.mean.astype(np.float32),
      "std": self.std.astype(np.float32),
      "eps": self.eps,
      "clip": self.clip,
      "preserve_tail_dim": self.preserve_tail_dim,
    }

  def normalize(self, value) -> np.ndarray:
    arr = to_numpy_detached(value)
    out = (arr - self.mean) / np.maximum(self.std, self.eps)
    out = np.clip(out, -self.clip, self.clip)
    return out.astype(np.float32)


def to_numpy_detached(value, dtype=np.float32) -> np.ndarray:
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  return np.asarray(value, dtype=dtype)


def detach_feature_dict(features: Optional[dict]) -> dict:
  if not features:
    return {}
  out = {}
  for key, value in features.items():
    if value is None:
      continue
    out[key] = to_numpy_detached(value)
  return out


def mean_pool_h(h_in) -> np.ndarray:
  h = to_numpy_detached(h_in)
  if h.ndim == 3:
    assert h.shape[0] == 1, f"expected single batch h_in, got {h.shape}"
    h = h[0]
  assert h.ndim == 2, f"h_in must be (Q,H) or (1,Q,H), got {h.shape}"
  h_feat = h.mean(axis=0).astype(np.float32)
  assert np.isfinite(h_feat).all(), "h_feat contains NaN/Inf"
  return h_feat


def summarize_base_chunk(
  base_chunk,
  summary_type: str = "selected_steps",
  selected_steps: tuple[int, ...] = (0, 5, 10, 20, 29),
) -> np.ndarray:
  chunk = to_numpy_detached(base_chunk)
  assert chunk.ndim == 2 and chunk.shape[1] == ACTION_DIM, f"base_chunk shape {chunk.shape}"
  if summary_type == "selected_steps":
    indices = [min(max(int(idx), 0), chunk.shape[0] - 1) for idx in selected_steps]
    summary = chunk[indices].reshape(-1)
  elif summary_type == "first_last_delta_mean":
    first = chunk[0]
    last = chunk[-1]
    summary = np.concatenate([first, last, last - first, chunk.mean(axis=0)], axis=0)
  elif summary_type == "flatten_full_chunk":
    summary = chunk.reshape(-1)
  else:
    raise ValueError(f"unsupported chunk summary type: {summary_type}")
  summary = summary.astype(np.float32)
  assert np.isfinite(summary).all(), "base chunk summary contains NaN/Inf"
  return summary


def flatten_proprio(proprio) -> np.ndarray:
  proprio_np = to_numpy_detached(proprio)
  proprio_np = proprio_np.reshape(-1).astype(np.float32)
  assert np.isfinite(proprio_np).all(), "proprio contains NaN/Inf"
  return proprio_np


def build_actor_obs(
  h_in,
  proprio,
  base_chunk,
  a_ref,
  action_normalizer: Optional[ActionNormalizer],
  summary_type: str = "selected_steps",
  selected_steps: tuple[int, ...] = (0, 5, 10, 20, 29),
) -> tuple[np.ndarray, dict]:
  normalizer = action_normalizer or ActionNormalizer.identity()
  h_feat = mean_pool_h(h_in)
  proprio_flat = flatten_proprio(proprio)
  chunk_summary = summarize_base_chunk(base_chunk, summary_type, selected_steps)
  a_ref_norm = normalizer.normalize(a_ref)
  actor_obs = np.concatenate([h_feat, proprio_flat, chunk_summary, a_ref_norm], axis=0).astype(np.float32)
  assert np.isfinite(actor_obs).all(), "actor_obs contains NaN/Inf"
  meta = {
    "h_feat_shape": tuple(h_feat.shape),
    "proprio_shape": tuple(proprio_flat.shape),
    "base_chunk_summary_shape": tuple(chunk_summary.shape),
    "a_ref_norm_shape": tuple(a_ref_norm.shape),
    "actor_obs_shape": tuple(actor_obs.shape),
    "ref_obs_slice": (actor_obs.shape[0] - ACTION_DIM, actor_obs.shape[0]),
  }
  return actor_obs, meta


def make_raw_fields(
  h_in,
  h_preout,
  proprio,
  base_chunk,
  a_ref,
  priv_state,
  a_exec,
  reward,
  done,
  success,
  timeout,
  next_h_in,
  next_h_preout,
  next_proprio,
  next_base_chunk,
  next_a_ref,
  next_priv_state,
) -> dict:
  fields = {
    "h_in": to_numpy_detached(h_in),
    "proprio": to_numpy_detached(proprio),
    "base_chunk": to_numpy_detached(base_chunk),
    "a_ref": to_numpy_detached(a_ref),
    "priv_state": to_numpy_detached(priv_state),
    "a_exec": to_numpy_detached(a_exec),
    "reward": np.asarray(float(reward), dtype=np.float32),
    "done": np.asarray(float(done), dtype=np.float32),
    "success": np.asarray(bool(success)),
    "timeout": np.asarray(bool(timeout)),
    "next_h_in": to_numpy_detached(next_h_in),
    "next_proprio": to_numpy_detached(next_proprio),
    "next_base_chunk": to_numpy_detached(next_base_chunk),
    "next_a_ref": to_numpy_detached(next_a_ref),
    "next_priv_state": to_numpy_detached(next_priv_state),
  }
  if h_preout is not None:
    fields["h_preout"] = to_numpy_detached(h_preout)
  if next_h_preout is not None:
    fields["next_h_preout"] = to_numpy_detached(next_h_preout)
  return fields


def make_fast_fields(
  raw_fields: dict,
  action_normalizer: Optional[ActionNormalizer],
  summary_type: str = "selected_steps",
  selected_steps: tuple[int, ...] = (0, 5, 10, 20, 29),
  feature_hook: str = "traj_decoder_input",
  actor_obs_normalizer: Optional[VectorNormalizer] = None,
  critic_obs_normalizer: Optional[VectorNormalizer] = None,
) -> tuple[dict, dict]:
  normalizer = action_normalizer or ActionNormalizer.identity()
  h_key = "h_preout" if feature_hook == "pre_output" and "h_preout" in raw_fields else "h_in"
  next_h_key = "next_h_preout" if feature_hook == "pre_output" and "next_h_preout" in raw_fields else "next_h_in"
  actor_obs, meta = build_actor_obs(
    raw_fields[h_key],
    raw_fields["proprio"],
    raw_fields["base_chunk"],
    raw_fields["a_ref"],
    normalizer,
    summary_type,
    selected_steps,
  )
  next_actor_obs, next_meta = build_actor_obs(
    raw_fields[next_h_key],
    raw_fields["next_proprio"],
    raw_fields["next_base_chunk"],
    raw_fields["next_a_ref"],
    normalizer,
    summary_type,
    selected_steps,
  )
  critic_obs = to_numpy_detached(raw_fields["priv_state"]).reshape(-1).astype(np.float32)
  next_critic_obs = to_numpy_detached(raw_fields["next_priv_state"]).reshape(-1).astype(np.float32)
  if actor_obs_normalizer is not None:
    actor_obs = actor_obs_normalizer.normalize(actor_obs)
    next_actor_obs = actor_obs_normalizer.normalize(next_actor_obs)
  if critic_obs_normalizer is not None:
    critic_obs = critic_obs_normalizer.normalize(critic_obs)
    next_critic_obs = critic_obs_normalizer.normalize(next_critic_obs)
  action_norm = normalizer.normalize(raw_fields["a_exec"])
  ref_action_norm = normalizer.normalize(raw_fields["a_ref"])
  fast = {
    "actor_obs": actor_obs,
    "critic_obs": critic_obs,
    "action": action_norm,
    "ref_action": ref_action_norm,
    "reward": np.asarray(float(raw_fields["reward"]), dtype=np.float32),
    "next_actor_obs": next_actor_obs,
    "next_critic_obs": next_critic_obs,
    "done": np.asarray(float(raw_fields["done"]), dtype=np.float32),
  }
  for key, value in fast.items():
    value_np = np.asarray(value)
    assert np.isfinite(value_np).all(), f"fast field {key} contains NaN/Inf"
  meta["next_actor_obs_shape"] = next_meta["actor_obs_shape"]
  meta["critic_obs_shape"] = tuple(critic_obs.shape)
  meta["next_critic_obs_shape"] = tuple(next_critic_obs.shape)
  meta["feature_hook"] = feature_hook
  return fast, meta
