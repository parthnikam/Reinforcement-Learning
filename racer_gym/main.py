import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch

try:
    from .agent import PPOConfig, RacingAgent
except ImportError:
    from agent import PPOConfig, RacingAgent


DEFAULT_MODEL_PATH = Path(__file__).with_name("racer_ppo.pt")
DEFAULT_EPISODES = 100
DEFAULT_MAX_STEPS = 4000
DEFAULT_AGENT_DELAY = 0.0
DEFAULT_SEED = None
DEFAULT_PPO_CONFIG = PPOConfig()


def make_env(
    render_mode: str | None,
    seed: int | None = None,
    domain_randomize: bool = False,
):
    env = gym.make(
        "CarRacing-v3",
        domain_randomize=domain_randomize,
        continuous=True,
        render_mode=render_mode,
    )
    if seed is not None:
        env.action_space.seed(seed)
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
            observation, _ = env.reset(seed=reset_seed)
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                if should_quit():
                    return

                action = get_player_action()
                observation, reward, terminated, truncated, _ = env.step(action)
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
            observation, _ = env.reset(seed=reset_seed)
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                if should_quit():
                    return

                action = agent.act(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
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
    load_existing: bool = True,
) -> RacingAgent:
    env = make_env(render_mode=None, seed=seed, domain_randomize=False)
    agent = RacingAgent(
        env,
        seed=seed,
        model_path=model_path,
        config=config,
        load_existing=load_existing,
    )

    try:
        agent.train(episodes=episodes, max_steps=max_steps, seed=seed)
    finally:
        env.close()

    return agent


def make_training_agent(
    agent_index: int,
    seed: int | None,
    model_path: Path,
    config: PPOConfig,
    load_existing: bool,
) -> tuple[RacingAgent, int | None]:
    agent_seed = None if seed is None else seed + agent_index
    env = make_env(render_mode=None, seed=agent_seed, domain_randomize=False)
    agent = RacingAgent(
        env,
        seed=agent_seed,
        model_path=model_path,
        config=config,
        load_existing=load_existing,
    )
    agent.model_path = None
    return agent, agent_seed


def average_agent_policy_states(agents: list[RacingAgent]) -> dict[str, torch.Tensor]:
    if not agents:
        raise ValueError("At least one agent is required for policy sharing.")

    state_dicts = [agent.policy.model.state_dict() for agent in agents]
    averaged_state = {}
    for key in state_dicts[0]:
        stacked_params = torch.stack([state[key].detach().float().cpu() for state in state_dicts])
        averaged_state[key] = stacked_params.mean(dim=0)
    return averaged_state


def share_agent_learnings(agents: list[RacingAgent]) -> None:
    averaged_state = average_agent_policy_states(agents)
    for agent in agents:
        device_state = {
            key: value.to(agent.policy.device)
            for key, value in averaged_state.items()
        }
        agent.policy.model.load_state_dict(device_state)
        agent.reset_optimizer()


def train_multiagent(
    num_agents: int,
    share_every: int,
    episodes: int,
    max_steps: int,
    seed: int | None,
    model_path: Path,
    config: PPOConfig,
    load_existing: bool = True,
) -> None:
    if num_agents < 2:
        raise ValueError("--num-agents must be at least 2 for --multiagent.")
    if share_every < 1:
        raise ValueError("--share-every must be at least 1.")

    agents_with_seeds = [
        make_training_agent(
            agent_index=index,
            seed=seed,
            model_path=model_path,
            config=config,
            load_existing=load_existing,
        )
        for index in range(num_agents)
    ]
    agents = [agent for agent, _ in agents_with_seeds]

    try:
        episodes_completed = 0
        while episodes_completed < episodes:
            round_episodes = min(share_every, episodes - episodes_completed)
            round_number = episodes_completed // share_every + 1
            print(f"Multi-agent round {round_number}: training {num_agents} agents for {round_episodes} episodes each")

            for index, (agent, agent_seed) in enumerate(agents_with_seeds, start=1):
                round_seed = None if agent_seed is None else agent_seed + episodes_completed
                print(f"Agent {index}/{num_agents} local training")
                agent.train(episodes=round_episodes, max_steps=max_steps, seed=round_seed)

            episodes_completed += round_episodes
            share_agent_learnings(agents)
            print(f"Shared policies after {episodes_completed} episodes per agent")

        if model_path is not None:
            agents[0].save(model_path)
            print(f"Saved shared PPO model to {model_path}")
    finally:
        for agent in agents:
            agent.env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play, watch, or train a CarRacing agent.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--play", action="store_true", help="Play CarRacing yourself.")
    mode.add_argument("--agent", action="store_true", help="Watch the current agent play.")
    mode.add_argument("--train", action="store_true", help="Train the agent.")
    mode.add_argument("--multiagent", action="store_true", help="Train multiple agents and periodically average their policies.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_PPO_CONFIG.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=DEFAULT_PPO_CONFIG.update_epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_CONFIG.learning_rate)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--share-every", type=int, default=5)
    parser.add_argument("--fresh", action="store_true", help="Train from new weights instead of loading an existing checkpoint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.play:
        run_player_game(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    elif args.agent:
        run_agent_game(
            episodes=args.episodes,
            max_steps=args.max_steps,
            delay=DEFAULT_AGENT_DELAY,
            seed=args.seed,
            model_path=args.model_path,
        )
    elif args.train:
        config = PPOConfig(
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        train_agent(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            model_path=args.model_path,
            config=config,
            load_existing=not args.fresh,
        )
    elif args.multiagent:
        config = PPOConfig(
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        train_multiagent(
            num_agents=args.num_agents,
            share_every=args.share_every,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            model_path=args.model_path,
            config=config,
            load_existing=not args.fresh,
        )


if __name__ == "__main__":
    main()
