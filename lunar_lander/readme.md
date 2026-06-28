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