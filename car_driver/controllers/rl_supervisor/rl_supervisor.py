import os, sys, subprocess
try: 
    import gymnasium as gym
except ImportError: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
    import gymnasium as gym

import numpy as np
import random
import math
from controller import Supervisor

class AltinoEndlessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # --- ACTION SPACE ---
        # 0: Up (Accelerate)
        # 1: Down (Decelerate/Reverse)
        # 2: Left (Steer Left)
        # 3: Right (Steer Right)
        # 4: Space (Brake)
        self.action_space = gym.spaces.Discrete(5)
        
        # --- OBSERVATION SPACE ---
        # What the AI sees: [Car X, Car Y, Car Yaw (rotation), Current Speed, Current Steering, Goal X, Goal Y]
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)
        
        # --- NODES ---
        self.vehicle = self.supervisor.getFromDef("VEHICLE")
        self.goal = self.supervisor.getFromDef("GOAL")
        
        # --- ENVIRONMENT SETTINGS ---
        self.speed = 0.0
        self.angle = 0.0
        self.MAX_SPEED = 1.8
        self.MAX_ANGLE = 0.4
        
        self.COLLECT_RADIUS = 0.3
        self.BOARD_LIMIT = 1.8 # For a 4x4 board (-2 to 2), we spawn within 1.8 to avoid walls
        
        self.step_count = 0
        self.max_steps = 2000 # End episode if it gets stuck for too long
        
        # Save start position for resets
        self.start_translation = self.vehicle.getField("translation").getSFVec3f()
        self.start_rotation = self.vehicle.getField("rotation").getSFRotation()

    def _respawn_goal(self):
        """Teleports the goal to a random X/Y coordinate on the board."""
        new_x = random.uniform(-self.BOARD_LIMIT, self.BOARD_LIMIT)
        new_y = random.uniform(-self.BOARD_LIMIT, self.BOARD_LIMIT)
        
        # Keep the original Z (height) so it doesn't sink into the floor
        current_pos = self.goal.getField("translation").getSFVec3f()
        self.goal.getField("translation").setSFVec3f([new_x, new_y, current_pos[2]])

    def _get_obs(self):
        """Gathers data for the neural network."""
        car_pos = self.vehicle.getField("translation").getSFVec3f()
        goal_pos = self.goal.getField("translation").getSFVec3f()
        
        # Extract the Yaw (Z-axis rotation) from the SFRotation field
        car_rot = self.vehicle.getField("rotation").getSFRotation()
        yaw = car_rot[3] if car_rot[2] > 0 else -car_rot[3] 
        
        return np.array([
            car_pos[0], car_pos[1], 
            yaw, 
            self.speed, self.angle, 
            goal_pos[0], goal_pos[1]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the world at the start of a new training episode."""
        super().reset(seed=seed)
        self.step_count = 0
        self.speed = 0.0
        self.angle = 0.0
        
        # Reset Vehicle
        self.vehicle.getField("translation").setSFVec3f(self.start_translation)
        self.vehicle.getField("rotation").setSFRotation(self.start_rotation)
        self.vehicle.resetPhysics()
        
        # Randomize the first goal
        self._respawn_goal()
        
        # Step simulation to apply physics resets
        self.supervisor.step(self.timestep)
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # --- 1. APPLY ACTION (Mimicking Human Keyboard Input) ---
        is_steering = False
        
        if action == 0:   # UP
            self.speed += 0.02
        elif action == 1: # DOWN
            self.speed -= 0.02
        elif action == 2: # LEFT
            self.angle -= 0.05
            is_steering = True
        elif action == 3: # RIGHT
            self.angle += 0.05
            is_steering = True
        elif action == 4: # SPACE
            self.speed = 0.0
            
        if not is_steering:
            self.angle = 0.0
            
        # Clamp values
        self.speed = max(min(self.speed, self.MAX_SPEED), -self.MAX_SPEED)
        self.angle = max(min(self.angle, self.MAX_ANGLE), -self.MAX_ANGLE)
        
        # Send to Altino via custom fields
        self.vehicle.getField("custom_steering").setSFFloat(float(self.angle))
        self.vehicle.getField("custom_speed").setSFFloat(float(self.speed))
        
        # Advance physics
        self.supervisor.step(self.timestep)
        
        # --- 2. CALCULATE REWARDS & COLLISIONS ---
        reward = -0.05 # Small time penalty to encourage moving fast
        terminated = False
        truncated = False
        
        car_pos = self.vehicle.getField("translation").getSFVec3f()
        goal_pos = self.goal.getField("translation").getSFVec3f()
        
        # Check Goal Collection
        dist_to_goal = math.sqrt((car_pos[0]-goal_pos[0])**2 + (car_pos[1]-goal_pos[1])**2)
        if dist_to_goal < self.COLLECT_RADIUS:
            reward += 20.0
            print(f"Goal Collected! Spawning new goal...", flush=True)
            self._respawn_goal()
            
        # Check Wall Collisions
        contact_points = self.vehicle.getContactPoints()
        for cp in contact_points:
            if cp.point[2] > 0.02: # Ignore tires on the floor
                node_id = getattr(cp, 'node_id', getattr(cp, 'nodeId', None))
                if node_id:
                    col_node = self.supervisor.getFromId(node_id)
                    while col_node:
                        if col_node.getDef() == "ARENA":
                            reward -= 5.0
                            terminated = True # End the episode if it crashes into a wall
                            print("Wall hit! Episode Terminated.", flush=True)
                            break
                        col_node = col_node.getParentNode()

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

# --- QUICK TEST SCRIPT ---
if __name__ == '__main__':
    # This block allows you to test the environment without launching the full RL trainer yet.
    # It will take random actions to ensure the spawning and walls work.
    print("--- SUPERVISOR ENV INITIATED ---", flush=True)
    env = AltinoEndlessEnv()
    obs, _ = env.reset()
    
    while True:
        # Take a completely random action (0 to 4)
        random_action = env.action_space.sample() 
        obs, reward, term, trunc, info = env.step(random_action)
        
        if term or trunc:
            env.reset()