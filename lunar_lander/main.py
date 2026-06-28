import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import trange
from gymnasium.envs.box2d.lunar_lander import FPS, LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W

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


class RandomTopStartWrapper(gym.Wrapper):
    def __init__(self, env, x_range: float = 0.85):
        super().__init__(env)
        self.x_range = x_range

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        core_env = self.env.unwrapped
        if not hasattr(core_env, "lander") or core_env.lander is None:
            return observation, info

        half_width = VIEWPORT_W / SCALE / 2
        center_x = half_width
        top_y = VIEWPORT_H / SCALE
        normalized_x = core_env.np_random.uniform(-self.x_range, self.x_range)
        target_x = center_x + normalized_x * half_width
        delta_x = target_x - core_env.lander.position.x
        delta_y = top_y - core_env.lander.position.y

        core_env.lander.position = (target_x, top_y)
        core_env.lander.linearVelocity = (
            core_env.np_random.uniform(-0.5, 0.5),
            core_env.np_random.uniform(-0.2, 0.2),
        )
        core_env.lander.angularVelocity = core_env.np_random.uniform(-0.05, 0.05)
        core_env.lander.angle = 0.0

        for leg in core_env.legs:
            leg.position = (leg.position.x + delta_x, leg.position.y + delta_y)
            leg.linearVelocity = core_env.lander.linearVelocity
            leg.angularVelocity = 0.0
            leg.ground_contact = False

        core_env.prev_shaping = None
        return self._get_observation(), info

    def _get_observation(self):
        core_env = self.env.unwrapped
        pos = core_env.lander.position
        vel = core_env.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (core_env.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            core_env.lander.angle,
            20.0 * core_env.lander.angularVelocity / FPS,
            1.0 if core_env.legs[0].ground_contact else 0.0,
            1.0 if core_env.legs[1].ground_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)


def make_lunar_lander(render_mode: str = "human", random_start: bool = True):
    try:
        env = gym.make(
            "LunarLander-v3",
            render_mode=render_mode,
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
        )
    except gym.error.VersionNotFound:
        env = gym.make("LunarLander-v2", render_mode=render_mode)
    except gym.error.DependencyNotInstalled as error:
        raise SystemExit(
            "LunarLander needs Box2D. Install it with:\n"
            '  pip install swig "gymnasium[box2d]"'
        ) from error

    if random_start:
        env = RandomTopStartWrapper(env)
    return env


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
