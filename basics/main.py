import argparse
from pathlib import Path

try:
    from .gridworld_env import GridWorldEnv
    from .play import run_player_game
    from .q_agent import GridWorldAgent
    from .training import run_visual_demo, train_gridworld
except ImportError:
    from gridworld_env import GridWorldEnv
    from play import run_player_game
    from q_agent import GridWorldAgent
    from training import run_visual_demo, train_gridworld


DEFAULT_MODEL_PATH = Path(__file__).with_name("gridworld_agent.pkl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, watch, or play GridWorld.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--play", action="store_true", help="Play GridWorld yourself.")
    mode.add_argument("--agent", action="store_true", help="Watch the trained agent play.")
    mode.add_argument("--train", action="store_true", help="Train the agent without demo playback.")

    parser.add_argument("--episodes", type=int, default=100, help="Training episodes.")
    parser.add_argument("--demo-episodes", type=int, default=4, help="Agent demo episodes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode.")
    parser.add_argument("--delay", type=float, default=0.20, help="Delay between agent moves.")
    parser.add_argument("--layout", type=int, default=None, help="Specific layout id to play/demo.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used to load/save the trained agent.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from an empty Q-table instead of loading the saved agent.",
    )
    return parser


def create_agent(env: GridWorldEnv) -> GridWorldAgent:
    return GridWorldAgent(
        env=env,
        learning_rate=0.5,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=0.998,
        final_epsilon=0.02,
    )


def main() -> None:
    args = build_parser().parse_args()
    env = GridWorldEnv(size=7)

    if args.play:
        run_player_game(env, max_steps=args.max_steps, layout_id=args.layout)
        return

    agent = create_agent(env)
    if not args.fresh and agent.load(args.model_path):
        print(f"Loaded saved agent from {args.model_path}")

    if args.episodes > 0:
        train_gridworld(
            env,
            agent,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            logging_interval=max(1, args.episodes // 32),
        )
        agent.save(args.model_path)
        print(f"Saved agent to {args.model_path}")

    if args.train:
        return

    run_visual_demo(
        env,
        agent,
        episodes=args.demo_episodes,
        delay=args.delay,
        max_steps=args.max_steps,
        layout_id=args.layout,
    )


if __name__ == "__main__":
    main()
