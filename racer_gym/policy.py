from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


ACTION_LOW = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
EPSILON = 1e-6


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ActorCriticCNN(nn.Module):
    def __init__(self, action_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.3, -0.7, -1.2], dtype=torch.float32))
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        nn.init.normal_(self.actor_mean.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

        # Steering starts centered, gas starts open, and brake starts nearly off.
        with torch.no_grad():
            self.actor_mean.bias.copy_(torch.tensor([0.0, 0.7, -2.0], dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        mean     = self.actor_mean(features)
        std      = torch.exp(self.log_std).expand_as(mean)
        value    = self.critic(features).squeeze(-1)
        return mean, std, value


class PPOPolicy:
    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device       = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model        = ActorCriticCNN().to(self.device)
        self.action_low   = ACTION_LOW.to(self.device)
        self.action_high  = ACTION_HIGH.to(self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias  = (self.action_high + self.action_low) / 2.0

    def choose_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.sample(observation_tensor, deterministic=deterministic)
        return output.action.squeeze(0).cpu().numpy().astype(np.float32)

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        mean, std, value = self.model(observations)
        distribution     = Normal(mean, std)
        raw_action       = mean if deterministic else distribution.rsample()
        squashed_action  = torch.tanh(raw_action)
        action           = squashed_action * self.action_scale + self.action_bias
        log_prob         = self._log_prob(distribution, raw_action, squashed_action)
        entropy          = distribution.entropy().sum(dim=-1)
        return PolicyOutput(action=action, log_prob=log_prob, entropy=entropy, value=value)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyOutput:
        mean, std, value = self.model(observations)
        distribution     = Normal(mean, std)
        squashed_action  = torch.clamp((actions - self.action_bias) / self.action_scale, -1.0 + EPSILON, 1.0 - EPSILON)
        raw_action       = torch.atanh(squashed_action)
        log_prob         = self._log_prob(distribution, raw_action, squashed_action)
        entropy          = distribution.entropy().sum(dim=-1)
        return PolicyOutput(action=actions, log_prob=log_prob, entropy=entropy, value=value)

    def save(self, path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _log_prob(
        self,
        distribution: Normal,
        raw_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + EPSILON)
        return log_prob.sum(dim=-1)
