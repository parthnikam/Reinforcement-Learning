from collections import defaultdict

import numpy as np

try:
    from .gridworld_env import GridWorldEnv
except ImportError:
    from gridworld_env import GridWorldEnv


class GridWorldAgent:
    def __init__(
        self,
        env: GridWorldEnv,
        learning_rate: float = 0.5,
        discount_factor: float = 0.95,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        final_epsilon: float = 0.05,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n, dtype=np.float32))
        self.episode_rewards = []
        self.episode_td_errors = []

    @staticmethod
    def state_to_key(observation: dict) -> tuple[int, ...]:
        agent_pos = tuple(int(x) for x in observation["agent"])
        target_pos = tuple(int(x) for x in observation["target"])
        bonuses = tuple(int(x) for x in observation["bonuses"])
        return agent_pos + target_pos + bonuses

    def get_action(self, state: tuple[int, ...], greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int, ...],
    ) -> float:
        future_q = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        target = reward + self.discount_factor * future_q
        td_error = target - float(self.q_table[state][action])
        self.q_table[state][action] += self.learning_rate * td_error
        self.episode_td_errors.append(td_error)
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
