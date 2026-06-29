# Lunar Lander RL Agent


```
pip install gymnasium
pip install tqdm
```


## Action Space
```
0: do nothing
1: fire left orientation engine
2: fire main engine
3: fire right orientation engine
```


## Observation Space: 

```Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)```

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.



## Play LunarLander yourself
```python main.py --play```

## Watch the policy agent play.
```python main.py --agent```

## Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train` | Flag | `False` | Train the policy agent. If omitted, the agent runs in evaluation mode. |
| `--episodes` | Integer | `1` | Number of episodes to run. |
| `--epochs` | Integer | `1000` | Number of training epochs for the policy. |
| `--max-steps` | Integer | `1000` | Maximum number of steps allowed in each episode before termination. |
| `--delay` | Float | `0.0` | Delay (in seconds) between agent actions for visualization. |
| `--seed` | Integer | `None` | Random seed for reproducible training and evaluation. |

### Example Usage

Train the agent:

```bash
python main.py --train --epochs 2000 --episodes 500
```

Run the trained agent:

```bash
python main.py --episodes 10
```

Run with a visualization delay:

```bash
python main.py --delay 0.1
```

Run with a fixed random seed:

```bash
python main.py --train --seed 42
```

## Policy (Heuristic)

This project includes a simple heuristic policy implemented in `policy.py` as `HeuristicLanderPolicy` using a `PolicyWeights` dataclass. The policy maps the 8-dimensional observation to one of four discrete actions (see Action Space above) using a set of tunable gains and thresholds.

### PolicyWeights (defaults)
- `x_gain`: 0.55
- `vx_gain`: 1.00
- `angle_gain`: 0.50
- `angular_velocity_gain`: 1.00
- `hover_x_gain`: 0.55
- `hover_y_gain`: 0.50
- `vy_gain`: 0.50
- `leg_vy_gain`: 0.50
- `main_threshold`: 0.05
- `side_threshold`: 0.05

### How the policy decides
- Input observation: `[x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]`.
- Compute a `target_angle` from `x` and `vx`, clipped to [-0.4, 0.4].
- Compute `hover_target` from the horizontal offset and `hover` gains.
- `angle_todo` combines angle error and angular velocity feedback.
- `hover_todo` combines vertical error and vertical velocity feedback.
- If either leg is in contact, the policy prioritizes stabilizing vertical velocity using `leg_vy_gain` and sets `angle_todo` to 0.
- Decision rules (returns action index):
	- If `hover_todo > abs(angle_todo)` and `hover_todo > main_threshold` → `2` (fire main engine)
	- Else if `angle_todo < -side_threshold` → `3` (fire right orientation engine)
	- Else if `angle_todo > side_threshold` → `1` (fire left orientation engine)
	- Otherwise → `0` (do nothing)

### API / Usage
Import and use the policy in code or from the command line runner. Examples:

```python
from lunar_lander.policy import HeuristicLanderPolicy, load_policy_or_default
import numpy as np

# Load saved policy or use defaults
policy = load_policy_or_default("lander_policy.json")

# Given an observation `obs` (8-d vector)
action = policy.choose_action(obs)

# Save current weights
policy.save("lander_policy.json")

# Create a perturbed (mutated) copy for parameter search
rng = np.random.default_rng(1234)
mutated = policy.mutated(rng, scale=0.02)
```

The default policy file used in this workspace is `lunar_lander/lander_policy.json` (a sample exists in the repository). You can tune the policy by editing that JSON file or by generating mutants with `mutated()` and evaluating them.

### Tips
- Start by adjusting `x_gain`, `vx_gain`, `angle_gain`, and `hover_y_gain` for coarse behavior changes.
- Use `mutated()` with small `scale` values to perform localized search of weight space.
