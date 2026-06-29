from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .cnn_pipe import preprocess_observation
    from .policy import PPOPolicy
except ImportError:
    from cnn_pipe import preprocess_observation
    from policy import PPOPolicy


@dataclass
class PPOConfig:
    rollout_steps: int = 1024
    update_epochs: int = 4
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5
    gas_reward_coef: float = 0.02
    brake_penalty_coef: float = 0.05
    idle_penalty: float = 0.05
    idle_gas_threshold: float = 0.1
    idle_brake_threshold: float = 0.1


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class RacingAgent:
    def __init__(
        self,
        env,
        policy: PPOPolicy | None = None,
        seed: int | None = None,
        model_path: Path | None = None,
        config: PPOConfig | None = None,
        load_existing: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy if policy is not None else PPOPolicy()
        self.config = config if config is not None else PPOConfig()
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=self.config.learning_rate)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.model_path = model_path
        if load_existing and model_path is not None and model_path.exists():
            self.load(model_path)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.policy.model.eval()
        processed_observation = preprocess_observation(observation)
        return self.policy.choose_action(processed_observation, deterministic=deterministic)

    def train(self, episodes: int, max_steps: int, seed: int | None = None) -> None:
        self.policy.model.train()
        buffer = RolloutBuffer()
        global_step = 0
        last_observation = None
        last_done = True

        for episode in range(1, episodes + 1):
            reset_seed = None if seed is None else seed + episode - 1
            observation, _ = self.env.reset(seed=reset_seed)
            total_reward = 0.0
            last_observation = observation
            last_done = False

            for step in range(1, max_steps + 1):
                processed_observation = preprocess_observation(observation)
                action, log_prob, value = self._sample_training_action(processed_observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                shaped_reward = self._shape_reward(float(reward), action)
                done = terminated or truncated
                total_reward += float(reward)
                global_step += 1

                buffer.add(
                    observation=processed_observation,
                    action=action,
                    log_prob=log_prob,
                    reward=shaped_reward,
                    done=done,
                    value=value,
                )

                observation = next_observation
                last_observation = observation
                last_done = done

                if len(buffer) >= self.config.rollout_steps:
                    self._update_from_rollout(buffer, next_observation=observation, done=done)
                    buffer.clear()

                if done:
                    print(f"Training episode {episode}: reward={total_reward:.2f}, steps={step}, total_steps={global_step}")
                    break
            else:
                print(f"Training episode {episode}: step limit reached, reward={total_reward:.2f}, total_steps={global_step}")

        if len(buffer) > 0 and last_observation is not None:
            self._update_from_rollout(buffer, next_observation=last_observation, done=last_done)

        if self.model_path is not None:
            self.save(self.model_path)
            print(f"Saved PPO model to {self.model_path}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(path)

    def load(self, path: Path) -> None:
        self.policy.load(path)

    def _shape_reward(self, reward: float, action: np.ndarray) -> float:
        gas = float(action[1])
        brake = float(action[2])
        idle_penalty = self.config.idle_penalty if (
            gas < self.config.idle_gas_threshold
            and brake < self.config.idle_brake_threshold
        ) else 0.0

        return (
            reward
            + self.config.gas_reward_coef * gas
            - self.config.brake_penalty_coef * brake
            - idle_penalty
        )

    def _sample_training_action(self, processed_observation: np.ndarray) -> tuple[np.ndarray, float, float]:
        observation_tensor = torch.as_tensor(
            processed_observation,
            dtype=torch.float32,
            device=self.policy.device,
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.policy.sample(observation_tensor, deterministic=False)

        action = output.action.squeeze(0).cpu().numpy().astype(np.float32)
        log_prob = float(output.log_prob.squeeze(0).cpu().item())
        value = float(output.value.squeeze(0).cpu().item())
        return action, log_prob, value

    def _update_from_rollout(
        self,
        buffer: RolloutBuffer,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        device = self.policy.device
        observations = torch.as_tensor(np.array(buffer.observations), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(buffer.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)
        values = torch.as_tensor(buffer.values, dtype=torch.float32, device=device)
        returns, advantages = self._compute_returns_and_advantages(buffer, next_observation, done)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = min(self.config.batch_size, len(buffer))
        indices = np.arange(len(buffer))

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(buffer), batch_size):
                batch_indices = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long, device=device)
                output = self.policy.evaluate_actions(
                    observations=observations[batch_indices],
                    actions=actions[batch_indices],
                )

                ratio = torch.exp(output.log_prob - old_log_probs[batch_indices])
                unclipped_policy_loss = -advantages[batch_indices] * ratio
                clipped_policy_loss = -advantages[batch_indices] * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
                value_loss = nn.functional.mse_loss(output.value, returns[batch_indices])
                entropy_loss = output.entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

    def _compute_returns_and_advantages(
        self,
        buffer: RolloutBuffer,
        next_observation: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        next_value = 0.0
        if not done:
            processed_observation = preprocess_observation(next_observation)
            observation_tensor = torch.as_tensor(
                processed_observation,
                dtype=torch.float32,
                device=self.policy.device,
            ).unsqueeze(0)
            with torch.no_grad():
                _, _, value = self.policy.model(observation_tensor)
            next_value = float(value.squeeze(0).cpu().item())

        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)
        values = np.array(buffer.values + [next_value], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0

        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * values[step + 1] * next_non_terminal - values[step]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_advantage
            advantages[step] = last_advantage

        returns = advantages + values[:-1]
        return returns, advantages
