import os
import sys
from typing import Optional

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


AGENT = "\u263b"
GOAL = "\u25e8"
REWARD = "\u25c8"
WALL = "\u22a0"
TOP_LEFT = "\u300c"
TOP_RIGHT = "\u300d"
BOTTOM_LEFT = "\ufe41"
BOTTOM_RIGHT = "\ufe42"
VERTICAL = "\u2503"
HORIZONTAL = "\u2501"


DEFAULT_LAYOUTS = [
    {
        "obstacles": {
            (1, 1), (1, 2), (1, 5),
            (2, 5),
            (3, 1), (3, 2), (3, 3),
            (4, 3),
            (5, 0), (5, 5),
        },
        "bonus_rewards": {
            (0, 5): 10,
            (2, 2): 10,
            (4, 5): 10,
            (6, 1): 10,
        },
    },
    {
        "obstacles": {
            (0, 3), (1, 1), (1, 3), (1, 5),
            (2, 1), (2, 5),
            (3, 1), (3, 2), (3, 3), (3, 5),
            (4, 5), (5, 0), (5, 2), (5, 3),
        },
        "bonus_rewards": {
            (0, 6): 10,
            (2, 3): 10,
            (4, 1): 10,
            (6, 5): 10,
        },
    },
    {
        "obstacles": {
            (1, 0), (1, 1), (1, 2), (1, 4),
            (2, 4), (2, 5),
            (3, 1), (3, 2), (3, 4),
            (4, 1), (4, 6),
            (5, 3), (5, 4), (5, 5),
        },
        "bonus_rewards": {
            (0, 2): 10,
            (2, 0): 10,
            (4, 4): 10,
            (6, 6): 10,
        },
    },
    {
        "obstacles": {
            (0, 1), (0, 5),
            (1, 3), (1, 5),
            (2, 1), (2, 3), (2, 5),
            (3, 1), (3, 5),
            (4, 1), (4, 3), (4, 5),
            (5, 1), (5, 3),
            (6, 3),
        },
        "bonus_rewards": {
            (0, 6): 10,
            (3, 3): 10,
            (5, 5): 10,
            (6, 0): 10,
        },
    },
]


class GridWorldEnv(gym.Env):
    def __init__(
        self,
        size: int = 7,
        obstacles: Optional[set[tuple[int, int]]] = None,
        bonus_rewards: Optional[dict[tuple[int, int], float]] = None,
        layouts: Optional[list[dict]] = None,
    ):
        self.size = size
        raw_layouts = layouts or DEFAULT_LAYOUTS
        if obstacles is not None or bonus_rewards is not None:
            raw_layouts = [
                {
                    "obstacles": obstacles or set(),
                    "bonus_rewards": bonus_rewards or {},
                }
            ]

        self.layouts = [self._sanitize_layout(layout) for layout in raw_layouts]
        self.layout_id = 0
        self.obstacles = set()
        self.bonus_rewards = {}
        self._all_bonus_locations = sorted(
            {
                location
                for layout in self.layouts
                for location in layout["bonus_rewards"]
            }
        )
        self._apply_layout(self.layout_id)
        self._remaining_bonus_locations = set(self.bonus_rewards)

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "bonuses": gym.spaces.MultiBinary(len(self._all_bonus_locations)),
                "layout_id": gym.spaces.Discrete(len(self.layouts)),
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

    def _in_bounds(self, location: tuple[int, int]) -> bool:
        row, col = location
        return 0 <= row < self.size and 0 <= col < self.size

    def _sanitize_layout(self, layout: dict) -> dict:
        obstacles = {
            location
            for location in layout.get("obstacles", set())
            if self._in_bounds(location)
        }
        bonus_rewards = {
            location: reward
            for location, reward in layout.get("bonus_rewards", {}).items()
            if self._in_bounds(location) and location not in obstacles
        }
        return {"obstacles": obstacles, "bonus_rewards": bonus_rewards}

    def _apply_layout(self, layout_id: int) -> None:
        self.layout_id = layout_id
        layout = self.layouts[self.layout_id]
        self.obstacles = set(layout["obstacles"])
        self.bonus_rewards = dict(layout["bonus_rewards"])

    def _get_obs(self):
        bonus_mask = np.array(
            [
                location in self._remaining_bonus_locations
                for location in self._all_bonus_locations
            ],
            dtype=np.int8,
        )
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "bonuses": bonus_mask,
            "layout_id": self.layout_id,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and "layout_id" in options:
            layout_id = int(options["layout_id"]) % len(self.layouts)
        else:
            layout_id = int(self.np_random.integers(0, len(self.layouts)))

        self._apply_layout(layout_id)
        self._remaining_bonus_locations = set(self.bonus_rewards)

        open_cells = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in self.obstacles
        ]

        agent_idx = int(self.np_random.integers(0, len(open_cells)))
        self._agent_location = np.array(open_cells[agent_idx], dtype=int)

        target_cells = [
            cell
            for cell in open_cells
            if cell != tuple(self._agent_location)
            and cell not in self._remaining_bonus_locations
        ]
        target_idx = int(self.np_random.integers(0, len(target_cells)))
        self._target_location = np.array(target_cells[target_idx], dtype=int)

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        next_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        bumped_wall = tuple(next_location) in self.obstacles

        if not bumped_wall:
            self._agent_location = next_location

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False

        if bumped_wall:
            reward = -10
        elif terminated:
            reward = 25
        else:
            reward = -1
            agent_cell = tuple(int(x) for x in self._agent_location)
            if agent_cell in self._remaining_bonus_locations:
                reward = self.bonus_rewards[agent_cell]
                self._remaining_bonus_locations.remove(agent_cell)

        info = self._get_info()
        info["bumped_wall"] = bumped_wall
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, total_reward: float = 0.0, step: int = 0, clear: bool = False):
        if clear:
            os.system("cls" if os.name == "nt" else "clear")

        board = np.full((self.size, self.size), " ", dtype="U1")
        agent_row, agent_col = self._agent_location
        target_row, target_col = self._target_location

        for row, col in self.obstacles:
            if self._in_bounds((row, col)):
                board[row, col] = WALL
        for row, col in self._remaining_bonus_locations:
            if self._in_bounds((row, col)):
                board[row, col] = REWARD

        if self._in_bounds((target_row, target_col)):
            board[target_row, target_col] = GOAL
        if self._in_bounds((agent_row, agent_col)):
            board[agent_row, agent_col] = AGENT

        print(f"GridWorld | step={step} | total_reward={total_reward:.2f}")
        print(TOP_LEFT + HORIZONTAL * (self.size * 3) + TOP_RIGHT)
        for row in board:
            cells = "".join(f" {cell} " for cell in row)
            print(f"{VERTICAL}{cells}{VERTICAL}")
        print(BOTTOM_LEFT + HORIZONTAL * (self.size * 3) + BOTTOM_RIGHT)
        print(f"{AGENT}=agent  {GOAL}=goal  {WALL}=wall  {REWARD}=reward\n")

    def close(self):
        return None
