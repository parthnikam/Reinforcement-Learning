import time

import numpy as np
from tqdm import trange

try:
    from .gridworld_env import GridWorldEnv
    from .q_agent import GridWorldAgent
except ImportError:
    from gridworld_env import GridWorldEnv
    from q_agent import GridWorldAgent


def train_gridworld(
    env: GridWorldEnv,
    agent: GridWorldAgent,
    n_episodes: int = 1000,
    max_steps: int = 50,
    logging_interval: int = 100,
) -> None:
    progress = trange(1, n_episodes + 1, desc="Training", unit="episode")
    for episode in progress:
        layout_count = len(getattr(env, "layouts", []))
        reset_options = None
        if layout_count:
            reset_options = {"layout_id": (episode - 1) % layout_count}

        observation, info = env.reset(options=reset_options)
        state = agent.state_to_key(observation)
        episode_reward = 0.0
        episode_tds = []

        for step in range(max_steps):
            action = agent.get_action(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.state_to_key(next_observation)
            td_error = agent.update(state, action, float(reward), terminated, next_state)
            episode_tds.append(td_error)
            episode_reward += float(reward)
            state = next_state

            if terminated or truncated:
                break

        agent.episode_rewards.append(episode_reward)
        agent.decay_epsilon()

        if episode % logging_interval == 0 or episode == n_episodes:
            avg_reward = float(np.mean(agent.episode_rewards[-logging_interval:]))
            avg_td = float(np.mean(episode_tds)) if episode_tds else 0.0
            progress.set_postfix(
                avg_reward=f"{avg_reward:.3f}",
                episode_td=f"{avg_td:.5f}",
                epsilon=f"{agent.epsilon:.3f}",
            )


def run_visual_demo(
    env: GridWorldEnv,
    agent: GridWorldAgent,
    episodes: int = 3,
    delay: float = 0.3,
    max_steps: int = 50,
    layout_id: int | None = None,
) -> None:
    print("\nRunning CLI GridWorld with greedy policy:\n")
    for demo_idx in range(1, episodes + 1):
        reset_options = None
        if layout_id is not None:
            reset_options = {"layout_id": layout_id}

        observation, info = env.reset(options=reset_options)
        state = agent.state_to_key(observation)
        total_reward = 0.0
        env.render(total_reward=total_reward, step=0, clear=True)
        print(f"Demo episode {demo_idx}/{episodes}")
        time.sleep(delay)

        steps = trange(1, max_steps + 1, desc=f"Episode {demo_idx}", unit="move")
        for step in steps:
            action = agent.get_action(state, greedy=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            state = agent.state_to_key(observation)
            env.render(total_reward=total_reward, step=step, clear=True)
            steps.set_postfix(reward=f"{reward:.2f}", total=f"{total_reward:.2f}")
            time.sleep(delay)

            if terminated or truncated:
                print(f"Reached the goal in {step} moves with reward {total_reward:.2f}.\n")
                break
        else:
            print(f"Step limit reached. Final reward: {total_reward:.2f}.\n")
