import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class PolicyWeights:
    x_gain: float = 0.55
    vx_gain: float = 1.00
    angle_gain: float = 0.50
    angular_velocity_gain: float = 1.00
    hover_x_gain: float = 0.55
    hover_y_gain: float = 0.50
    vy_gain: float = 0.50
    leg_vy_gain: float = 0.50
    main_threshold: float = 0.05
    side_threshold: float = 0.05


class HeuristicLanderPolicy:
    def __init__(self, weights: PolicyWeights | None = None):
        self.weights = weights or PolicyWeights()

    def choose_action(self, observation) -> int:
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
        weights = self.weights

        target_angle = x * weights.x_gain + vx * weights.vx_gain
        target_angle = float(np.clip(target_angle, -0.4, 0.4))
        hover_target = weights.hover_x_gain * abs(x)

        angle_todo = (
            (target_angle - angle) * weights.angle_gain
            - angular_velocity * weights.angular_velocity_gain
        )
        hover_todo = (
            (hover_target - y) * weights.hover_y_gain
            - vy * weights.vy_gain
        )

        if left_leg or right_leg:
            angle_todo = 0.0
            hover_todo = -vy * weights.leg_vy_gain

        if hover_todo > abs(angle_todo) and hover_todo > weights.main_threshold:
            return 2
        if angle_todo < -weights.side_threshold:
            return 3
        if angle_todo > weights.side_threshold:
            return 1
        return 0

    def mutated(self, rng: np.random.Generator, scale: float) -> "HeuristicLanderPolicy":
        values = asdict(self.weights)
        for key, value in values.items():
            values[key] = max(0.01, float(value + rng.normal(0.0, scale)))
        return HeuristicLanderPolicy(PolicyWeights(**values))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self.weights), file, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HeuristicLanderPolicy":
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            weights = PolicyWeights(**json.load(file))
        return cls(weights)


def load_policy_or_default(path: str | Path) -> HeuristicLanderPolicy:
    path = Path(path)
    if path.exists():
        return HeuristicLanderPolicy.load(path)
    return HeuristicLanderPolicy()
