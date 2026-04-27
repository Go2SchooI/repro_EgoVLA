from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


RL_MODES = (
  "collect_base",
  "offline_rl",
  "online_rl",
  "eval_rl",
  "eval_identity_actor",
  "eval_tiny_noise",
  "eval_residual_scale_sweep",
  "debug_trace_action_path",
)


def str2bool(value):
  if isinstance(value, bool):
    return value
  if value is None:
    return False
  value = str(value).strip().lower()
  if value in ("1", "true", "t", "yes", "y", "on"):
    return True
  if value in ("0", "false", "f", "no", "n", "off"):
    return False
  raise ValueError(f"invalid boolean value: {value}")


def _parse_int_list(value: Optional[object], default: Sequence[int]) -> list[int]:
  if value is None:
    return list(default)
  if isinstance(value, (list, tuple)):
    return [int(v) for v in value]
  text = str(value).strip()
  if not text:
    return list(default)
  return [int(v.strip()) for v in text.split(",") if v.strip()]


@dataclass
class RLConfig:
  enabled: bool = False
  mode: str = "collect_base"
  actor_insert_point: str = "after_temporal_smoothing"
  feature_hook: str = "traj_decoder_input"
  freeze_egovla: bool = True
  cache_rl_features: bool = True
  reward_type: str = "sparse_success"
  beta_ref: float = 1.0
  alpha_entropy: float = 0.2
  gamma: float = 0.99
  tau: float = 0.005
  batch_size: int = 256
  replay_capacity: int = 1_000_000
  ref_dropout_p: float = 0.0
  deterministic_eval: bool = True
  debug_dump_shapes: bool = False
  save_debug_transition_path: str = "debug_rl_transition.pt"
  replay_path: str = "rl_replay.pt"
  action_normalizer_path: Optional[str] = None
  actor_checkpoint_path: Optional[str] = None
  critic_checkpoint_path: Optional[str] = None
  save_rl_checkpoint_path: Optional[str] = None
  load_rl_checkpoint_path: Optional[str] = None
  chunk_summary_type: str = "selected_steps"
  chunk_summary_steps: tuple[int, ...] = (0, 5, 10, 20, 29)
  action_norm_clip: float = 0.999
  obs_norm_clip: float = 10.0
  lr: float = 3e-4
  update_steps: int = 1000
  updates_per_env_step: int = 1
  min_replay_size: int = 256
  max_debug_steps: int = 1
  hidden_dim: int = 256
  noise_scale: float = 0.0
  noise_type: str = "gaussian"
  noise_seed: int = 0
  residual_scale: float = 1.0
  action_diff_log_path: Optional[str] = None
  action_diff_step_start: int = -1
  action_diff_step_end: int = -1
  wandb_enabled: bool = False
  wandb_project: str = "egovla-rl-posttrain"
  wandb_entity: Optional[str] = None
  wandb_run_name: Optional[str] = None
  wandb_group: Optional[str] = None
  wandb_tags: tuple[str, ...] = ()

  def validate(self) -> None:
    if self.mode not in RL_MODES:
      raise ValueError(f"rl.mode={self.mode!r} must be one of {RL_MODES}")
    if self.actor_insert_point != "after_temporal_smoothing":
      raise ValueError("v1 only supports rl.actor_insert_point=after_temporal_smoothing")
    if self.feature_hook not in ("traj_decoder_input", "pre_output"):
      raise ValueError("rl.feature_hook must be traj_decoder_input or pre_output")
    if self.reward_type != "sparse_success":
      raise ValueError("v1 only supports rl.reward_type=sparse_success")
    if not 0.0 <= self.ref_dropout_p <= 1.0:
      raise ValueError("rl.ref_dropout_p must be in [0, 1]")
    if self.batch_size <= 0:
      raise ValueError("rl.batch_size must be positive")
    if self.replay_capacity <= 0:
      raise ValueError("rl.replay_capacity must be positive")
    if self.min_replay_size <= 0:
      self.min_replay_size = self.batch_size
    if self.noise_scale < 0:
      raise ValueError("rl.noise_scale must be non-negative")
    if self.noise_type not in ("gaussian", "uniform"):
      raise ValueError("rl.noise_type must be gaussian or uniform")
    if self.residual_scale < 0:
      raise ValueError("rl.residual_scale must be non-negative")
    if self.action_diff_step_end >= 0 and self.action_diff_step_start >= 0:
      if self.action_diff_step_end < self.action_diff_step_start:
        raise ValueError("rl.action_diff_step_end must be >= rl.action_diff_step_start")


def _add_arg(parser, flat_name: str, **kwargs) -> None:
  aliases = [f"--{flat_name}"]
  if flat_name.startswith("rl_"):
    aliases.append("--" + flat_name.replace("rl_", "rl.", 1))
  parser.add_argument(*aliases, dest=flat_name, **kwargs)


def add_rl_args(parser) -> None:
  _add_arg(parser, "rl_enabled", type=str2bool, default=False)
  _add_arg(parser, "rl_mode", type=str, default="collect_base", choices=RL_MODES)
  _add_arg(parser, "rl_actor_insert_point", type=str, default="after_temporal_smoothing")
  _add_arg(parser, "rl_feature_hook", type=str, default="traj_decoder_input")
  _add_arg(parser, "rl_freeze_egovla", type=str2bool, default=True)
  _add_arg(parser, "rl_cache_rl_features", type=str2bool, default=True)
  _add_arg(parser, "rl_reward_type", type=str, default="sparse_success")
  _add_arg(parser, "rl_beta_ref", type=float, default=1.0)
  _add_arg(parser, "rl_alpha_entropy", type=float, default=0.2)
  _add_arg(parser, "rl_gamma", type=float, default=0.99)
  _add_arg(parser, "rl_tau", type=float, default=0.005)
  _add_arg(parser, "rl_batch_size", type=int, default=256)
  _add_arg(parser, "rl_replay_capacity", type=int, default=1_000_000)
  _add_arg(parser, "rl_ref_dropout_p", type=float, default=0.0)
  _add_arg(parser, "rl_deterministic_eval", type=str2bool, default=True)
  _add_arg(parser, "rl_debug_dump_shapes", type=str2bool, default=False)
  _add_arg(parser, "rl_save_debug_transition_path", type=str, default="debug_rl_transition.pt")
  _add_arg(parser, "rl_replay_path", type=str, default="rl_replay.pt")
  _add_arg(parser, "rl_action_normalizer_path", type=str, default=None)
  _add_arg(parser, "rl_actor_checkpoint_path", type=str, default=None)
  _add_arg(parser, "rl_critic_checkpoint_path", type=str, default=None)
  _add_arg(parser, "rl_save_rl_checkpoint_path", type=str, default=None)
  _add_arg(parser, "rl_load_rl_checkpoint_path", type=str, default=None)
  _add_arg(parser, "rl_chunk_summary_type", type=str, default="selected_steps")
  _add_arg(parser, "rl_chunk_summary_steps", type=str, default="0,5,10,20,29")
  _add_arg(parser, "rl_action_norm_clip", type=float, default=0.999)
  _add_arg(parser, "rl_obs_norm_clip", type=float, default=10.0)
  _add_arg(parser, "rl_lr", type=float, default=3e-4)
  _add_arg(parser, "rl_update_steps", type=int, default=1000)
  _add_arg(parser, "rl_updates_per_env_step", type=int, default=1)
  _add_arg(parser, "rl_min_replay_size", type=int, default=256)
  _add_arg(parser, "rl_max_debug_steps", type=int, default=1)
  _add_arg(parser, "rl_hidden_dim", type=int, default=256)
  _add_arg(parser, "rl_noise_scale", type=float, default=0.0)
  _add_arg(parser, "rl_noise_type", type=str, default="gaussian")
  _add_arg(parser, "rl_noise_seed", type=int, default=0)
  _add_arg(parser, "rl_residual_scale", type=float, default=1.0)
  _add_arg(parser, "rl_action_diff_log_path", type=str, default=None)
  _add_arg(parser, "rl_action_diff_step_start", type=int, default=-1)
  _add_arg(parser, "rl_action_diff_step_end", type=int, default=-1)
  _add_arg(parser, "rl_wandb_enabled", type=str2bool, default=False)
  _add_arg(parser, "rl_wandb_project", type=str, default="egovla-rl-posttrain")
  _add_arg(parser, "rl_wandb_entity", type=str, default=None)
  _add_arg(parser, "rl_wandb_run_name", type=str, default=None)
  _add_arg(parser, "rl_wandb_group", type=str, default=None)
  _add_arg(parser, "rl_wandb_tags", type=str, default="")


def build_rl_config(args) -> RLConfig:
  cfg = RLConfig(
    enabled=bool(getattr(args, "rl_enabled", False)),
    mode=getattr(args, "rl_mode", "collect_base"),
    actor_insert_point=getattr(args, "rl_actor_insert_point", "after_temporal_smoothing"),
    feature_hook=getattr(args, "rl_feature_hook", "traj_decoder_input"),
    freeze_egovla=bool(getattr(args, "rl_freeze_egovla", True)),
    cache_rl_features=bool(getattr(args, "rl_cache_rl_features", True)),
    reward_type=getattr(args, "rl_reward_type", "sparse_success"),
    beta_ref=float(getattr(args, "rl_beta_ref", 1.0)),
    alpha_entropy=float(getattr(args, "rl_alpha_entropy", 0.2)),
    gamma=float(getattr(args, "rl_gamma", 0.99)),
    tau=float(getattr(args, "rl_tau", 0.005)),
    batch_size=int(getattr(args, "rl_batch_size", 256)),
    replay_capacity=int(getattr(args, "rl_replay_capacity", 1_000_000)),
    ref_dropout_p=float(getattr(args, "rl_ref_dropout_p", 0.0)),
    deterministic_eval=bool(getattr(args, "rl_deterministic_eval", True)),
    debug_dump_shapes=bool(getattr(args, "rl_debug_dump_shapes", False)),
    save_debug_transition_path=getattr(args, "rl_save_debug_transition_path", "debug_rl_transition.pt"),
    replay_path=getattr(args, "rl_replay_path", "rl_replay.pt"),
    action_normalizer_path=getattr(args, "rl_action_normalizer_path", None),
    actor_checkpoint_path=getattr(args, "rl_actor_checkpoint_path", None),
    critic_checkpoint_path=getattr(args, "rl_critic_checkpoint_path", None),
    save_rl_checkpoint_path=getattr(args, "rl_save_rl_checkpoint_path", None),
    load_rl_checkpoint_path=getattr(args, "rl_load_rl_checkpoint_path", None),
    chunk_summary_type=getattr(args, "rl_chunk_summary_type", "selected_steps"),
    chunk_summary_steps=tuple(_parse_int_list(getattr(args, "rl_chunk_summary_steps", None), (0, 5, 10, 20, 29))),
    action_norm_clip=float(getattr(args, "rl_action_norm_clip", 0.999)),
    obs_norm_clip=float(getattr(args, "rl_obs_norm_clip", 10.0)),
    lr=float(getattr(args, "rl_lr", 3e-4)),
    update_steps=int(getattr(args, "rl_update_steps", 1000)),
    updates_per_env_step=int(getattr(args, "rl_updates_per_env_step", 1)),
    min_replay_size=int(getattr(args, "rl_min_replay_size", 256)),
    max_debug_steps=int(getattr(args, "rl_max_debug_steps", 1)),
    hidden_dim=int(getattr(args, "rl_hidden_dim", 256)),
    noise_scale=float(getattr(args, "rl_noise_scale", 0.0)),
    noise_type=getattr(args, "rl_noise_type", "gaussian"),
    noise_seed=int(getattr(args, "rl_noise_seed", 0)),
    residual_scale=float(getattr(args, "rl_residual_scale", 1.0)),
    action_diff_log_path=getattr(args, "rl_action_diff_log_path", None),
    action_diff_step_start=int(getattr(args, "rl_action_diff_step_start", -1)),
    action_diff_step_end=int(getattr(args, "rl_action_diff_step_end", -1)),
    wandb_enabled=bool(getattr(args, "rl_wandb_enabled", False)),
    wandb_project=getattr(args, "rl_wandb_project", "egovla-rl-posttrain"),
    wandb_entity=getattr(args, "rl_wandb_entity", None),
    wandb_run_name=getattr(args, "rl_wandb_run_name", None),
    wandb_group=getattr(args, "rl_wandb_group", None),
    wandb_tags=tuple(tag.strip() for tag in str(getattr(args, "rl_wandb_tags", "") or "").split(",") if tag.strip()),
  )
  cfg.validate()
  return cfg
