import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import trange

try:
    from .policy import HeuristicLanderPolicy, load_policy_or_default
except ImportError:
    from policy import HeuristicLanderPolicy, load_policy_or_default


ACTION_NAMES = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine",
}

DEFAULT_POLICY_PATH = Path(__file__).with_name("lander_policy.json")


def make_lunar_lander(render_mode: str = "human"):
    try:
        return gym.make("LunarLander-v3", render_mode=render_mode, continuous=False, gravity=-5.0,
                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    except gym.error.VersionNotFound:
        return gym.make("LunarLander-v2", render_mode=render_mode)
    except gym.error.DependencyNotInstalled as error:
        raise SystemExit(
            "LunarLander needs Box2D. Install it with:\n"
            '  pip install swig "gymnasium[box2d]"'
        ) from error


def choose_agent_action(observation, policy: HeuristicLanderPolicy) -> int:
    return policy.choose_action(observation)


def run_policy_episode(env, policy: HeuristicLanderPolicy, max_steps: int) -> float:
    observation, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = choose_agent_action(observation, policy)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    return total_reward


def train_policy(
    epochs: int,
    max_steps: int,
    policy_path: Path,
    seed: int | None = None,
) -> HeuristicLanderPolicy:
    rng = np.random.default_rng(seed)
    env = make_lunar_lander(render_mode=None)
    best_policy = load_policy_or_default(policy_path)
    best_score = run_policy_episode(env, best_policy, max_steps)

    progress = trange(1, epochs + 1, desc="Training policy", unit="epoch")
    try:
        for epoch in progress:
            mutation_scale = max(0.02, 0.30 * (1.0 - epoch / max(1, epochs)))
            candidate = best_policy.mutated(rng, scale=mutation_scale)
            score = run_policy_episode(env, candidate, max_steps)

            if score > best_score:
                best_policy = candidate
                best_score = score
                best_policy.save(policy_path)

            progress.set_postfix(best=f"{best_score:.1f}", latest=f"{score:.1f}")
    finally:
        env.close()

    best_policy.save(policy_path)
    print(f"Saved best policy to {policy_path} with score {best_score:.2f}")
    return best_policy


def get_player_action() -> int:
    import pygame

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]:
        return 2
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 3
    return 0


def run_player_game(episodes: int, max_steps: int) -> None:
    import pygame

    env = make_lunar_lander(render_mode="human")
    print("Controls: W/Up/Space = main engine, A/Left = left engine, D/Right = right engine, Esc/Q = quit")

    try:
        for episode in range(1, episodes + 1):
            observation, info = env.reset()
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_q}:
                        return

                action = get_player_action()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if terminated or truncated:
                    print(f"Episode {episode}: reward={total_reward:.2f}, steps={step}")
                    break
            else:
                print(f"Episode {episode}: step limit reached, reward={total_reward:.2f}")
    finally:
        env.close()


def run_agent_game(
    episodes: int,
    max_steps: int,
    delay: float,
    policy_path: Path,
) -> None:
    env = make_lunar_lander(render_mode="human")
    policy = load_policy_or_default(policy_path)

    try:
        for episode in range(1, episodes + 1):
            observation, info = env.reset()
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                action = choose_agent_action(observation, policy)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play or watch LunarLander.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--play", action="store_true", help="Play LunarLander yourself.")
    mode.add_argument("--agent", action="store_true", help="Watch the policy agent play.")
    mode.add_argument("--train", action="store_true", help="Train the policy agent.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--epochs", type=int, default=1000, help="Policy training epochs.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between agent steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for training.")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path used to load/save policy weights.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.play:
        run_player_game(episodes=args.episodes, max_steps=args.max_steps)
    elif args.agent:
        run_agent_game(
            episodes=args.episodes,
            max_steps=args.max_steps,
            delay=args.delay,
            policy_path=args.policy_path,
        )
    elif args.train:
        train_policy(
            epochs=args.epochs,
            max_steps=args.max_steps,
            policy_path=args.policy_path,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
