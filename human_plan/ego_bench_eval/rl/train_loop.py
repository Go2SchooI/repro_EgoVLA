from __future__ import annotations

import math
import argparse
from pathlib import Path

import torch

from .action_space import ACTION_DIM, ActionNormalizer
from .config import RLConfig, add_rl_args, build_rl_config
from .features import VectorNormalizer
from .replay_buffer import RLReplayBuffer
from .sac import SACRefBC, save_split_checkpoints


def _replay_needs_action_rebuild(replay: RLReplayBuffer, cfg: RLConfig) -> bool:
  state = replay.metadata.get("action_normalizer")
  if not state or state.get("mode") != "minmax_clip":
    return True

  max_abs = 0.0
  for transition in replay.transitions:
    max_abs = max(max_abs, float(torch.as_tensor(transition["fast"]["ref_action"]).abs().max()))
    max_abs = max(max_abs, float(torch.as_tensor(transition["fast"]["action"]).abs().max()))
    if max_abs > cfg.action_norm_clip + 1e-4:
      return True
  return False


def _replay_needs_obs_rebuild(replay: RLReplayBuffer, cfg: RLConfig) -> bool:
  actor_state = replay.metadata.get("actor_obs_normalizer")
  critic_state = replay.metadata.get("critic_obs_normalizer")
  if not actor_state or not critic_state:
    return True
  if abs(float(actor_state.get("clip", 10.0)) - float(cfg.obs_norm_clip)) > 1e-6:
    return True
  if abs(float(critic_state.get("clip", 10.0)) - float(cfg.obs_norm_clip)) > 1e-6:
    return True
  if int(actor_state.get("preserve_tail_dim", 0)) != ACTION_DIM:
    return True
  return False


def _maybe_init_wandb(cfg: RLConfig, replay: RLReplayBuffer):
  if not cfg.wandb_enabled:
    return None
  try:
    import wandb
  except ImportError as exc:
    raise ImportError("wandb is not installed in this environment; disable --rl_wandb_enabled or install wandb") from exc

  run = wandb.init(
    project=cfg.wandb_project,
    entity=cfg.wandb_entity,
    name=cfg.wandb_run_name,
    group=cfg.wandb_group,
    tags=list(cfg.wandb_tags),
    config={
      "rl": vars(cfg),
      "replay_size": len(replay),
      "actor_obs_dim": replay.actor_obs_dim,
      "critic_obs_dim": replay.critic_obs_dim,
      "action_dim": replay.action_dim,
      "replay_path": cfg.replay_path,
    },
  )
  return run


def offline_rl_from_config(cfg: RLConfig, device: torch.device | str) -> dict:
  replay = RLReplayBuffer.load(cfg.replay_path)
  if len(replay) == 0:
    raise RuntimeError(f"empty replay buffer: {cfg.replay_path}")

  normalizer = ActionNormalizer.from_state(replay.metadata.get("action_normalizer"))
  needs_action_rebuild = _replay_needs_action_rebuild(replay, cfg)
  needs_obs_rebuild = _replay_needs_obs_rebuild(replay, cfg)
  if needs_action_rebuild or needs_obs_rebuild:
    print(
      "offline_rl_rebuilding_fast_fields "
      f"path={cfg.replay_path} action_rebuild={needs_action_rebuild} "
      f"obs_rebuild={needs_obs_rebuild} action_mode=minmax_clip "
      f"action_clip={cfg.action_norm_clip} obs_clip={cfg.obs_norm_clip}",
      flush=True,
    )
    if needs_action_rebuild:
      normalizer = replay.fit_action_normalizer(clip=cfg.action_norm_clip)
    actor_obs_normalizer, critic_obs_normalizer = replay.fit_obs_normalizers(
      normalizer,
      cfg.chunk_summary_type,
      cfg.chunk_summary_steps,
      cfg.feature_hook,
      clip=cfg.obs_norm_clip,
    )
    replay.rebuild_fast_fields(
      normalizer,
      cfg.chunk_summary_type,
      cfg.chunk_summary_steps,
      cfg.feature_hook,
      actor_obs_normalizer,
      critic_obs_normalizer,
    )
    replay.save(cfg.replay_path)
    if cfg.action_normalizer_path:
      action_norm_path = Path(cfg.action_normalizer_path)
      if str(action_norm_path.parent) not in ("", "."):
        action_norm_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(normalizer.state_dict(), action_norm_path)
  else:
    actor_obs_normalizer = VectorNormalizer.from_state(replay.metadata.get("actor_obs_normalizer"))
    critic_obs_normalizer = VectorNormalizer.from_state(replay.metadata.get("critic_obs_normalizer"))

  if cfg.load_rl_checkpoint_path:
    agent, ckpt_normalizer, ckpt_metadata = SACRefBC.load(cfg.load_rl_checkpoint_path, cfg, device)
    if ckpt_normalizer is not None:
      normalizer = ckpt_normalizer
    actor_obs_normalizer = VectorNormalizer.from_state(
      ckpt_metadata.get("actor_obs_normalizer")
    ) or actor_obs_normalizer
    critic_obs_normalizer = VectorNormalizer.from_state(
      ckpt_metadata.get("critic_obs_normalizer")
    ) or critic_obs_normalizer
  else:
    ref_slice = replay.metadata.get("fast_field_shapes", {}).get("ref_obs_slice")
    ref_slice = tuple(ref_slice) if ref_slice is not None else None
    agent = SACRefBC(
      actor_obs_dim=replay.actor_obs_dim,
      critic_obs_dim=replay.critic_obs_dim,
      action_dim=replay.action_dim,
      cfg=cfg,
      device=device,
      ref_obs_slice=ref_slice,
    )

  wandb_run = _maybe_init_wandb(cfg, replay)
  metrics = {}
  steps = int(cfg.update_steps)
  for step in range(steps):
    batch = replay.sample_batch(cfg.batch_size, device)
    metrics = agent.update(batch)
    if wandb_run is not None:
      wandb_run.log(
        {
          **{f"offline_rl/{key}": value for key, value in metrics.items()},
          "offline_rl/update_step": step + 1,
          "offline_rl/replay_size": len(replay),
        },
        step=step + 1,
      )
    if step == 0 or (step + 1) == steps or (step + 1) % max(1, steps // 10) == 0:
      pretty = " ".join(f"{k}={v:.5g}" for k, v in metrics.items())
      print(f"offline_rl_update step={step + 1}/{steps} {pretty}", flush=True)
      if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError(f"non-finite SAC metrics at step {step + 1}: {metrics}")

  save_path = cfg.save_rl_checkpoint_path or cfg.load_rl_checkpoint_path or "rl_checkpoint.pt"
  agent.save(
    save_path,
    action_normalizer=normalizer,
    actor_obs_normalizer=actor_obs_normalizer,
    critic_obs_normalizer=critic_obs_normalizer,
    metadata={
      "replay_path": str(cfg.replay_path),
      "replay_size": len(replay),
      "replay_metadata": replay.metadata,
    },
  )
  save_split_checkpoints(agent, cfg.actor_checkpoint_path, cfg.critic_checkpoint_path)
  if wandb_run is not None:
    wandb_run.summary["checkpoint_path"] = str(save_path)
    wandb_run.finish()
  print(f"offline_rl_saved checkpoint={save_path}", flush=True)
  return metrics


def main() -> None:
  parser = argparse.ArgumentParser(description="Offline SAC-style RL post-training from a cached EgoVLA replay buffer.")
  parser.add_argument("--device", type=str, default="cuda:0")
  add_rl_args(parser)
  args = parser.parse_args()
  cfg = build_rl_config(args)
  if not cfg.enabled or cfg.mode != "offline_rl":
    raise ValueError("train_loop.py expects --rl_enabled true --rl_mode offline_rl")
  offline_rl_from_config(cfg, torch.device(args.device))


if __name__ == "__main__":
  main()
