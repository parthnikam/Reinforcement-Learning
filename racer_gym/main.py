import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

try:
    from .agent import PPOConfig, RacingAgent
except ImportError:
    from agent import PPOConfig, RacingAgent


DEFAULT_MODEL_PATH = Path(__file__).with_name("racer_ppo.pt")
DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 4000
DEFAULT_AGENT_DELAY = 0.0
DEFAULT_SEED = None
DEFAULT_PPO_CONFIG = PPOConfig()


def make_env(render_mode: str | None, seed: int | None = None):
    env = gym.make(
        "CarRacing-v3",
        domain_randomize=True,
        continuous=True,
        render_mode=render_mode,
    )
    return env


def get_player_action() -> np.ndarray:
    keys = pygame.key.get_pressed()
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        action[0] = -1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        action[0] = 1.0
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        action[1] = 1.0
    if keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_SPACE]:
        action[2] = 1.0

    return action


def should_quit() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_q}:
            return True
    return False


def run_player_game(episodes: int, max_steps: int, seed: int | None) -> None:
    pygame.init()
    pygame.display.init()
    env = make_env(render_mode="human", seed=seed)
    clock = pygame.time.Clock()

    print("Controls: A/Left steer left, D/Right steer right, W/Up gas, S/Down/Space brake, Esc/Q quit")
    try:
        for episode in range(1, episodes + 1):
            reset_seed = None if seed is None else seed + episode - 1
            observation, info = env.reset(seed=reset_seed)
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                if should_quit():
                    return

                action = get_player_action()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                clock.tick(60)

                if terminated or truncated:
                    print(f"Player episode {episode}: reward={total_reward:.2f}, steps={step}")
                    break
            else:
                print(f"Player episode {episode}: step limit reached, reward={total_reward:.2f}")
    finally:
        env.close()
        pygame.quit()


def run_agent_game(
    episodes: int,
    max_steps: int,
    delay: float,
    seed: int | None,
    model_path: Path,
) -> None:
    env = make_env(render_mode="human", seed=seed)
    agent = RacingAgent(env, seed=seed, model_path=model_path)

    try:
        for episode in range(1, episodes + 1):
            reset_seed = None if seed is None else seed + episode - 1
            observation, info = env.reset(seed=reset_seed)
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                if should_quit():
                    return

                action = agent.act(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if delay > 0:
                    time.sleep(delay)

                if terminated or truncated:
                    print(f"Agent episode {episode}: reward={total_reward:.2f}, steps={step}")
                    break
            else:
                print(f"Agent episode {episode}: step limit reached, reward={total_reward:.2f}")
    finally:
        env.close()


def train_agent(
    episodes: int,
    max_steps: int,
    seed: int | None,
    model_path: Path,
    config: PPOConfig,
) -> RacingAgent:
    env = make_env(render_mode=None, seed=seed)
    agent = RacingAgent(env, seed=seed, model_path=model_path, config=config)

    try:
        agent.train(episodes=episodes, max_steps=max_steps, seed=seed)
    finally:
        env.close()

    return agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play, watch, or train a CarRacing agent.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--play", action="store_true", help="Play CarRacing yourself.")
    mode.add_argument("--agent", action="store_true", help="Watch the current agent play.")
    mode.add_argument("--train", action="store_true", help="Train the agent.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.play:
        run_player_game(
            episodes=DEFAULT_EPISODES,
            max_steps=DEFAULT_MAX_STEPS,
            seed=DEFAULT_SEED,
        )
    elif args.agent:
        run_agent_game(
            episodes=DEFAULT_EPISODES,
            max_steps=DEFAULT_MAX_STEPS,
            delay=DEFAULT_AGENT_DELAY,
            seed=DEFAULT_SEED,
            model_path=DEFAULT_MODEL_PATH,
        )
    elif args.train:
        train_agent(
            episodes=DEFAULT_EPISODES,
            max_steps=DEFAULT_MAX_STEPS,
            seed=DEFAULT_SEED,
            model_path=DEFAULT_MODEL_PATH,
            config=DEFAULT_PPO_CONFIG,
        )


if __name__ == "__main__":
    main()
