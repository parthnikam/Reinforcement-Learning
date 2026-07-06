import argparse
import time
import os
import random
import numpy as np
import gymnasium as gym

try:
	import torch
	from torch import nn
	TORCH_AVAILABLE = True
except Exception:
	TORCH_AVAILABLE = False

try:
	import msvcrt
	HAS_MSVCRT = True
except Exception:
	HAS_MSVCRT = False


def load_agent(path):
	if not TORCH_AVAILABLE:
		print('Torch not available — falling back to random agent')
		return None
	if not os.path.exists(path):
		print(f'Agent file not found: {path}')
		return None
	try:
		obj = torch.load(path, map_location='cpu')
	except Exception as e:
		print('Failed to load agent with torch:', e)
		return None

	# If it's an nn.Module
	if isinstance(obj, nn.Module):
		model = obj
		model.eval()

		def policy(obs):
			with torch.no_grad():
				t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
				out = model(t)
				a = out.squeeze(0).cpu().numpy()
			return a

		return policy

	# If it's a dict with 'policy' or 'model'
	if isinstance(obj, dict):
		# try 'policy' callable
		if 'policy' in obj and callable(obj['policy']):
			return obj['policy']
		# try state_dict on a simple linear policy
		if 'state_dict' in obj and 'model_class' in obj:
			try:
				ModelClass = obj['model_class']
				model = ModelClass()
				model.load_state_dict(obj['state_dict'])
				model.eval()

				def policy(obs):
					with torch.no_grad():
						t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
						out = model(t)
						return out.squeeze(0).cpu().numpy()

				return policy
			except Exception:
				pass

	print('Loaded object unsupported — falling back to random agent')
	return None


def human_controller(env, prev_action, scale=0.6):
	# Poll simple keyboard controls and map to continuous actions.
	# Keys: w/s forward/back, a/d left/right, q to quit, space to zero
	action = np.zeros(env.action_space.shape, dtype=float)
	forward = 0.0
	side = 0.0
	if HAS_MSVCRT and msvcrt.kbhit():
		ch = msvcrt.getwch()
		if ch in ('w', 'W'):
			forward = 1.0
		elif ch in ('s', 'S'):
			forward = -1.0
		elif ch in ('a', 'A'):
			side = -1.0
		elif ch in ('d', 'D'):
			side = 1.0
		elif ch in (' ',):
			return np.zeros_like(action), False
		elif ch in ('q', 'Q'):
			return prev_action, True

	# Map directional intents into action vector.
	# Ant has 8 actuators; we'll distribute directional commands across them.
	if env.action_space.shape[0] >= 4:
		half = env.action_space.shape[0] // 2
		action[:half] = forward * scale
		action[half:half * 2] = side * scale
	else:
		action[:] = (forward + side) * scale

	return action, False


def run(env_name='Walker2d-v5', ctrl_cost_weight=0.5, play=False, agent_path=None, max_steps=10000):
	env = gym.make(env_name, ctrl_cost_weight=ctrl_cost_weight, render_mode='human')
	obs, info = env.reset()

	policy = None
	if agent_path:
		policy = load_agent(agent_path)

	step = 0
	prev_action = np.zeros(env.action_space.shape)
	print('Controls: w/s/a/d to steer, space to stop, q to quit (human mode)')

	while step < max_steps:
		if play:
			action, should_quit = human_controller(env, prev_action)
			if should_quit:
				break
		else:
			if policy is not None:
				try:
					action = policy(obs)
				except Exception:
					action = env.action_space.sample()
			else:
				action = env.action_space.sample()

		obs, reward, terminated, truncated, info = env.step(action)
		prev_action = action
		step += 1
		done = terminated or truncated
		if done:
			obs, info = env.reset()

	env.close()


def main():
	parser = argparse.ArgumentParser(description='Play Ant-v5 human or agent')
	parser.add_argument('--play', action='store_true', help='Enable human play mode')
	parser.add_argument('--agent', type=str, default=None, help='Path to agent .pt file (optional)')
	parser.add_argument('--max-steps', type=int, default=10000, help='Maximum steps to run')
	args = parser.parse_args()

	run(play=args.play, agent_path=args.agent, max_steps=args.max_steps)


if __name__ == '__main__':
	main()



