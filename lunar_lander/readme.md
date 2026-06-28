# Lunar Lander RL Agent


pip install gymnasium
pip install tqdm


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