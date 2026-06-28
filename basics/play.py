try:
    from .gridworld_env import GridWorldEnv
except ImportError:
    from gridworld_env import GridWorldEnv


ACTION_BY_INPUT = {
    "d": 0,
    "right": 0,
    "w": 1,
    "up": 1,
    "a": 2,
    "left": 2,
    "s": 3,
    "down": 3,
}


def run_player_game(
    env: GridWorldEnv,
    max_steps: int = 100,
    layout_id: int | None = None,
) -> None:
    reset_options = None
    if layout_id is not None:
        reset_options = {"layout_id": layout_id}

    observation, info = env.reset(options=reset_options)
    total_reward = 0.0
    env.render(total_reward=total_reward, step=0, clear=True)
    print("Move with W/A/S/D or type up/down/left/right. Type q to quit.")

    for step in range(1, max_steps + 1):
        raw_action = input("move> ").strip().lower()
        if raw_action in {"q", "quit", "exit"}:
            print(f"Quit game. Final reward: {total_reward:.2f}")
            return

        if raw_action not in ACTION_BY_INPUT:
            env.render(total_reward=total_reward, step=step - 1, clear=True)
            print("Invalid move. Use W/A/S/D, up/down/left/right, or q.")
            continue

        observation, reward, terminated, truncated, info = env.step(
            ACTION_BY_INPUT[raw_action]
        )
        total_reward += float(reward)
        env.render(total_reward=total_reward, step=step, clear=True)

        if info.get("bumped_wall"):
            print("Wall hit: -10")
        else:
            print(f"Reward this move: {reward:.2f}")

        if terminated or truncated:
            print(f"Reached the goal in {step} moves. Final reward: {total_reward:.2f}")
            return

    print(f"Step limit reached. Final reward: {total_reward:.2f}")
