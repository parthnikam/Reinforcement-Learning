try:
    from .gridworld_env import GridWorldEnv
    from .q_agent import GridWorldAgent
    from .training import run_visual_demo, train_gridworld
except ImportError:
    from gridworld_env import GridWorldEnv
    from q_agent import GridWorldAgent
    from training import run_visual_demo, train_gridworld


def main() -> None:
    env = GridWorldEnv(size=7)
    agent = GridWorldAgent(
        env=env,
        learning_rate=0.5,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.05,
    )

    train_gridworld(env, agent, n_episodes=1000, max_steps=50, logging_interval=100)
    run_visual_demo(env, agent, episodes=3, delay=0.25, max_steps=50)


if __name__ == "__main__":
    main()
