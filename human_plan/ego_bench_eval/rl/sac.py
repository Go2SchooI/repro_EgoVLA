from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .action_space import ActionNormalizer
from .actor_critic import DoubleQCritic, GaussianActor, soft_update
from .config import RLConfig
from .features import VectorNormalizer


class SACRefBC:
  def __init__(
    self,
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    cfg: RLConfig,
    device: torch.device | str,
    ref_obs_slice: Optional[tuple[int, int]] = None,
  ):
    self.cfg = cfg
    self.device = torch.device(device)
    self.actor_obs_dim = int(actor_obs_dim)
    self.critic_obs_dim = int(critic_obs_dim)
    self.action_dim = int(action_dim)
    self.ref_obs_slice = ref_obs_slice
    self.actor = GaussianActor(actor_obs_dim, action_dim, cfg.hidden_dim).to(self.device)
    self.critic = DoubleQCritic(critic_obs_dim, action_dim, cfg.hidden_dim).to(self.device)
    self.critic_target = DoubleQCritic(critic_obs_dim, action_dim, cfg.hidden_dim).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
    self.update_step = 0

  def _actor_obs_for_train(self, actor_obs: torch.Tensor) -> torch.Tensor:
    if self.cfg.ref_dropout_p <= 0.0 or self.ref_obs_slice is None:
      return actor_obs
    start, end = self.ref_obs_slice
    out = actor_obs.clone()
    mask = torch.rand((out.shape[0], 1), device=out.device) < self.cfg.ref_dropout_p
    out[:, start:end] = torch.where(mask, torch.zeros_like(out[:, start:end]), out[:, start:end])
    return out

  def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    actor_obs = batch["actor_obs"]
    critic_obs = batch["critic_obs"]
    action = batch["action"]
    ref_action = batch["ref_action"]
    reward = batch["reward"]
    next_actor_obs = batch["next_actor_obs"]
    next_critic_obs = batch["next_critic_obs"]
    done = batch["done"]

    with torch.no_grad():
      next_action, next_logp, _, _ = self.actor.sample(next_actor_obs)
      target_q1, target_q2 = self.critic_target(next_critic_obs, next_action)
      target_min_q = torch.minimum(target_q1, target_q2)
      target = reward + self.cfg.gamma * (1.0 - done) * (
        target_min_q - self.cfg.alpha_entropy * next_logp
      )

    q1, q2 = self.critic(critic_obs, action)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    self.critic_opt.zero_grad(set_to_none=True)
    q_loss.backward()
    self.critic_opt.step()

    train_actor_obs = self._actor_obs_for_train(actor_obs)
    pi_action, logp, _, _ = self.actor.sample(train_actor_obs)
    q1_pi, q2_pi = self.critic(critic_obs, pi_action)
    min_q_pi = torch.minimum(q1_pi, q2_pi)
    ref_loss_per = (pi_action - ref_action).pow(2).mean(dim=-1, keepdim=True)
    actor_loss = (
      self.cfg.alpha_entropy * logp
      - min_q_pi
      + self.cfg.beta_ref * ref_loss_per
    ).mean()
    self.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    self.actor_opt.step()

    soft_update(self.critic, self.critic_target, self.cfg.tau)
    self.update_step += 1

    with torch.no_grad():
      actor_ref_abs = (pi_action - ref_action).abs().mean()
      dataset_action_ref_abs = (action - ref_action).abs().mean()
      metrics = {
        "q_loss": float(q_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "ref_loss": float(ref_loss_per.mean().detach().cpu()),
        "log_prob": float(logp.mean().detach().cpu()),
        "entropy": float((-logp).mean().detach().cpu()),
        "mean_abs_ref_action_norm": float(ref_action.abs().mean().detach().cpu()),
        "mean_abs_dataset_action_minus_ref_norm": float(dataset_action_ref_abs.detach().cpu()),
        "mean_abs_actor_minus_ref_norm": float(actor_ref_abs.detach().cpu()),
        "mean_abs_action_ref": float(dataset_action_ref_abs.detach().cpu()),
        "mean_abs_a_exec_minus_a_ref_norm": float(actor_ref_abs.detach().cpu()),
        "q1_mean": float(q1.mean().detach().cpu()),
        "q2_mean": float(q2.mean().detach().cpu()),
      }
    return metrics

  @torch.no_grad()
  def act(self, actor_obs_np, deterministic: bool = False):
    self.actor.eval()
    return self.actor.act_np(actor_obs_np, self.device, deterministic=deterministic)

  def checkpoint(
    self,
    action_normalizer: Optional[ActionNormalizer] = None,
    actor_obs_normalizer: Optional[VectorNormalizer] = None,
    critic_obs_normalizer: Optional[VectorNormalizer] = None,
    metadata: Optional[dict] = None,
  ) -> dict:
    return {
      "actor_obs_dim": self.actor_obs_dim,
      "critic_obs_dim": self.critic_obs_dim,
      "action_dim": self.action_dim,
      "ref_obs_slice": self.ref_obs_slice,
      "cfg": vars(self.cfg).copy(),
      "actor": self.actor.state_dict(),
      "critic": self.critic.state_dict(),
      "critic_target": self.critic_target.state_dict(),
      "actor_opt": self.actor_opt.state_dict(),
      "critic_opt": self.critic_opt.state_dict(),
      "update_step": self.update_step,
      "action_normalizer": action_normalizer.state_dict() if action_normalizer is not None else None,
      "actor_obs_normalizer": actor_obs_normalizer.state_dict() if actor_obs_normalizer is not None else None,
      "critic_obs_normalizer": critic_obs_normalizer.state_dict() if critic_obs_normalizer is not None else None,
      "metadata": metadata or {},
    }

  def save(
    self,
    path: str | Path,
    action_normalizer: Optional[ActionNormalizer] = None,
    actor_obs_normalizer: Optional[VectorNormalizer] = None,
    critic_obs_normalizer: Optional[VectorNormalizer] = None,
    metadata: Optional[dict] = None,
  ) -> None:
    path = Path(path)
    if path.parent and str(path.parent) not in ("", "."):
      path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      self.checkpoint(
        action_normalizer=action_normalizer,
        actor_obs_normalizer=actor_obs_normalizer,
        critic_obs_normalizer=critic_obs_normalizer,
        metadata=metadata,
      ),
      path,
    )

  @classmethod
  def load(cls, path: str | Path, cfg: RLConfig, device: torch.device | str) -> tuple["SACRefBC", Optional[ActionNormalizer], dict]:
    try:
      ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
      ckpt = torch.load(path, map_location=device)
    agent = cls(
      ckpt["actor_obs_dim"],
      ckpt["critic_obs_dim"],
      ckpt["action_dim"],
      cfg,
      device,
      ref_obs_slice=tuple(ckpt["ref_obs_slice"]) if ckpt.get("ref_obs_slice") is not None else None,
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
    if "actor_opt" in ckpt:
      agent.actor_opt.load_state_dict(ckpt["actor_opt"])
    if "critic_opt" in ckpt:
      agent.critic_opt.load_state_dict(ckpt["critic_opt"])
    agent.update_step = int(ckpt.get("update_step", 0))
    normalizer = ActionNormalizer.from_state(ckpt.get("action_normalizer"))
    metadata = dict(ckpt.get("metadata", {}))
    metadata["actor_obs_normalizer"] = ckpt.get("actor_obs_normalizer")
    metadata["critic_obs_normalizer"] = ckpt.get("critic_obs_normalizer")
    return agent, normalizer, metadata


def save_split_checkpoints(
  agent: SACRefBC,
  actor_path: Optional[str],
  critic_path: Optional[str],
) -> None:
  if actor_path:
    path = Path(actor_path)
    if path.parent and str(path.parent) not in ("", "."):
      path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), path)
  if critic_path:
    path = Path(critic_path)
    if path.parent and str(path.parent) not in ("", "."):
      path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.critic.state_dict(), path)
