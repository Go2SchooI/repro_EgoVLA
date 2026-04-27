from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
  dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
  layers = []
  for in_dim, out_dim in zip(dims[:-1], dims[1:]):
    layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
  layers.append(nn.Linear(dims[-1], int(output_dim)))
  return nn.Sequential(*layers)


class GaussianActor(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.action_dim = int(action_dim)
    self.net = mlp(obs_dim, (hidden_dim, hidden_dim), 2 * action_dim)

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = self.net(obs)
    mu, log_std = torch.chunk(out, 2, dim=-1)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std

  def sample(
    self,
    obs: torch.Tensor,
    deterministic: bool = False,
    eps: float = 1e-6,
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    mu, log_std = self.forward(obs)
    std = log_std.exp()
    if deterministic:
      u = mu
      action = torch.tanh(u)
      return action, None, mu, log_std
    normal = torch.distributions.Normal(mu, std)
    u = normal.rsample()
    action = torch.tanh(u)
    log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + eps)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob, mu, log_std

  @torch.no_grad()
  def act_np(self, obs_np: np.ndarray, device: torch.device | str, deterministic: bool = False) -> np.ndarray:
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).reshape(1, -1)
    action, _, _, _ = self.sample(obs, deterministic=deterministic)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


class QNetwork(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
    super().__init__()
    self.net = mlp(obs_dim + action_dim, (hidden_dim, hidden_dim), 1)

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat([obs, action], dim=-1))


class DoubleQCritic(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.action_dim = int(action_dim)
    self.q1 = QNetwork(obs_dim, action_dim, hidden_dim)
    self.q2 = QNetwork(obs_dim, action_dim, hidden_dim)

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.q1(obs, action), self.q2(obs, action)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
  with torch.no_grad():
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
      tgt_param.data.mul_(1.0 - tau)
      tgt_param.data.add_(tau * src_param.data)
