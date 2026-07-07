import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from controller import Supervisor

class WebotsGoalEnv(gym.Env):
    def __init__(self):
        super(WebotsGoalEnv, self).__init__()
        
        # Initialize Webots Supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Get Node References
        self.vehicle = self.supervisor.getFromDef("VEHICLE")
        self.goal = self.supervisor.getFromDef("GOAL")
        
        # Verify nodes exist before proceeding
        if self.vehicle is None:
            raise RuntimeError("CRITICAL: 'VEHICLE' node not found. Ensure you converted it to a Base Node.")
        if self.goal is None:
            raise RuntimeError("CRITICAL: 'GOAL' node not found. Check your DEF name in the Scene Tree.")
            
        # Field pointers for resetting
        self.vehicle_translation = self.vehicle.getField("translation")
        self.vehicle_rotation = self.vehicle.getField("rotation")
        self.goal_translation = self.goal.getField("translation")
        
        # Capture original vehicle properties to ensure clean resets
        self.initial_vehicle_pos = self.vehicle_translation.getSFVec3f()
        self.initial_vehicle_rot = self.vehicle_rotation.getSFRotation()
        
        # --- ADJUST ENVIRONMENT CONFIGURATION HERE ---
        self.ARENA_LIMIT = 0.9          # Geofence boundary (half-width of your arena)
        self.COLLECT_RADIUS = 0.15      # Distance to trigger goal collection
        self.MAX_STEPS = 500            # Max actions per episode
        self.current_step = 0
        
        # --- PENALTY CONFIGURATION ---
        self.MIN_SPEED_THRESHOLD = 0.15 # Speeds below this are treated as "not taking action"
        self.IDLE_PENALTY = -0.1       # Penalty per step for staying stationary or idling
        
        # Define Action Space: [Speed, Angle]
        # Speed: [-1.8 to 1.8], Angle: [-0.4 to 0.4]
        self.action_space = spaces.Box(
            low=np.array([-1.8, -0.4], dtype=np.float32),
            high=np.array([1.8, 0.4], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define Observation Space: [Car_X, Car_Y, Goal_X, Goal_Y]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_obs(self):
        car_pos = self.vehicle_translation.getSFVec3f()
        goal_pos = self.goal_translation.getSFVec3f()
        # Return [Car_X, Car_Y, Goal_X, Goal_Y]
        return np.array([car_pos[0], car_pos[1], goal_pos[0], goal_pos[1]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Reset Vehicle back to center with original Z height & rotation
        self.vehicle_translation.setSFVec3f([0.0, 0.0, self.initial_vehicle_pos[2]])
        self.vehicle_rotation.setSFRotation(self.initial_vehicle_rot)
        self.vehicle.resetPhysics() # Stop all physical momentum
        
        # 2. Spawn Goal at a random location within arena boundaries
        new_x = random.uniform(-self.ARENA_LIMIT + 0.1, self.ARENA_LIMIT - 0.1)
        new_y = random.uniform(-self.ARENA_LIMIT + 0.1, self.ARENA_LIMIT - 0.1)
        current_goal_pos = self.goal_translation.getSFVec3f()
        self.goal_translation.setSFVec3f([new_x, new_y, current_goal_pos[2]])
        
        # Clear customData so car doesn't immediately move on reset
        self.vehicle.getField("customData").setSFString("0.0,0.0")
        
        # Advance 1 physics step to apply resets in Webots
        self.supervisor.step(self.timestep)
        
        self.current_step = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        
        # 1. Write actions to customData
        speed = float(action[0])
        angle = float(action[1])
        self.vehicle.getField("customData").setSFString(f"{speed},{angle}")
        
        # 2. Run simulation step
        self.supervisor.step(self.timestep)
        
        # 3. Retrieve new state
        obs = self._get_obs()
        car_x, car_y, goal_x, goal_y = obs
        
        # 4. Calculate distance to goal
        dist_to_goal = math.sqrt((car_x - goal_x)**2 + (car_y - goal_y)**2)
        
        # Base living penalty to encourage finding the fastest path
        reward = -0.01  
        terminated = False
        truncated = False
        
        # --- Check "Not Taking Action" (Idling) Penalty ---
        if abs(speed) < self.MIN_SPEED_THRESHOLD:
            reward += self.IDLE_PENALTY  # Add penalty for staying stationary
            
        # --- Check Termination Conditions & Rewards ---
        
        # A. Hit Walls (Geofencing)
        if abs(car_x) > self.ARENA_LIMIT or abs(car_y) > self.ARENA_LIMIT:
            reward = -1.0
            terminated = True
            print(f"[Episode End] Hit Wall! Step: {self.current_step}", flush=True)
            
        # B. Reached Goal
        elif dist_to_goal < self.COLLECT_RADIUS:
            reward = 10.0
            terminated = True
            print(f"[Episode End] Goal Reached! Step: {self.current_step}", flush=True)
            
        # C. Timeout (Too many steps)
        elif self.current_step >= self.MAX_STEPS:
            reward = -1.0
            truncated = True
            print(f"[Episode End] Timeout! Max actions reached.", flush=True)
            
        info = {}
        return obs, reward, terminated, truncated, info

# --- Training Loop Execution ---
if __name__ == "__main__":
    print("Initializing Webots Gym Environment...", flush=True)
    env = WebotsGoalEnv()
    
    print("Starting PPO Agent Training...", flush=True)
    # Instantiate PPO model with tensorboard logging
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log="./ppo_logs/")
    
    # Train the model (e.g., 50,000 simulator steps)
    model.learn(total_timesteps=50000)
    
    # Save the trained policy
    model.save("ppo_altino_model")
    print("Training Complete! Model saved.", flush=True)