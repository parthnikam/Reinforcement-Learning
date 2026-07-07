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

        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.vehicle = self.supervisor.getFromDef("VEHICLE")
        self.goal = self.supervisor.getFromDef("GOAL")

        if self.vehicle is None:
            raise RuntimeError("CRITICAL: 'VEHICLE' node not found.")
        if self.goal is None:
            raise RuntimeError("CRITICAL: 'GOAL' node not found.")

        self.vehicle_translation = self.vehicle.getField("translation")
        self.vehicle_rotation = self.vehicle.getField("rotation")
        self.goal_translation = self.goal.getField("translation")

        self.initial_vehicle_pos = self.vehicle_translation.getSFVec3f()
        self.initial_vehicle_rot = self.vehicle_rotation.getSFRotation()

        self.emitter = self.supervisor.getDevice("emitter")
        if self.emitter is None:
            raise RuntimeError("CRITICAL: 'emitter' device not found.")

        # --- ENV CONFIG ---
        self.ARENA_LIMIT = 0.9
        self.COLLECT_RADIUS = 0.15
        self.MAX_STEPS = 500
        self.current_step = 0

        self.MIN_SPEED_THRESHOLD = 0.15
        self.IDLE_PENALTY = -0.1

        # Curriculum: start goal close, expand radius as agent improves
        self.MIN_GOAL_DIST = 0.3
        self.curriculum_radius = 0.35      # starting spawn radius
        self.max_curriculum_radius = self.ARENA_LIMIT - 0.1
        self.episodes_done = 0
        self.CURRICULUM_GROWTH_EVERY = 50  # widen radius every N episodes
        self.CURRICULUM_STEP = 0.05

        self.prev_dist = None  # for potential-based shaping

        # Action space unchanged: [Speed, Angle]
        self.action_space = spaces.Box(
            low=np.array([-1.8, -0.4], dtype=np.float32),
            high=np.array([1.8, 0.4], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: [rel_x, rel_y, dist, sin(heading), cos(heading), sin(bearing_to_goal), cos(bearing_to_goal)]
        # Egocentric + heading, instead of raw absolute coordinates.
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_heading(self):
        # Webots rotation field is axis-angle [x, y, z, angle]; for a car rotating
        # about Z this angle IS the yaw (adjust axis check if your model differs).
        rot = self.vehicle_rotation.getSFRotation()
        axis_z, angle = rot[2], rot[3]
        yaw = angle if axis_z >= 0 else -angle
        return yaw

    def _get_obs(self):
        car_pos = self.vehicle_translation.getSFVec3f()
        goal_pos = self.goal_translation.getSFVec3f()
        yaw = self._get_heading()

        dx = goal_pos[0] - car_pos[0]
        dy = goal_pos[1] - car_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        bearing_to_goal = math.atan2(dy, dx) - yaw  # angle to goal relative to car's facing
        bearing_to_goal = math.atan2(math.sin(bearing_to_goal), math.cos(bearing_to_goal))  # normalize to [-pi, pi]

        self._last_dist = dist
        self._last_abs_bearing = abs(bearing_to_goal)
        return np.array([
            dx, dy, dist,
            math.sin(yaw), math.cos(yaw),
            math.sin(bearing_to_goal), math.cos(bearing_to_goal)
        ], dtype=np.float32)

    def _spawn_goal(self):
        # Curriculum-based spawn: random angle, radius grows over training
        radius = random.uniform(self.MIN_GOAL_DIST, self.curriculum_radius)
        if random.random() < 0.4:
            theta = random.choice([math.pi/2, -math.pi/2]) + random.uniform(-0.3, 0.3)
        else:
            theta = random.uniform(0, 2 * math.pi)

        new_x = radius * math.cos(theta)
        new_y = radius * math.sin(theta)
        new_x = max(-self.ARENA_LIMIT + 0.1, min(self.ARENA_LIMIT - 0.1, new_x))
        new_y = max(-self.ARENA_LIMIT + 0.1, min(self.ARENA_LIMIT - 0.1, new_y))
        current_goal_pos = self.goal_translation.getSFVec3f()
        self.goal_translation.setSFVec3f([new_x, new_y, current_goal_pos[2]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Fully stop the vehicle BEFORE moving it, so no stale velocity carries over
        self.emitter.send("0.0,0.0".encode('utf-8'))
        self.supervisor.step(self.timestep)  # let stop command land on the controller

        # 2. Reset transform + zero all physics (linear & angular velocity)
        self.vehicle_translation.setSFVec3f([0.0, 0.0, self.initial_vehicle_pos[2]])
        self.vehicle_rotation.setSFRotation(self.initial_vehicle_rot)
        self.vehicle.resetPhysics()

        # 3. Respawn goal using curriculum radius, guaranteed not to overlap start
        self._spawn_goal()

        # 4. Let a few physics steps pass so nothing is mid-collision/mid-fall
        #    at the moment the agent takes its first action of the new episode
        for _ in range(3):
            self.emitter.send("0.0,0.0".encode('utf-8'))
            self.supervisor.step(self.timestep)

        self.current_step = 0
        self.episodes_done += 1
        if self.episodes_done % self.CURRICULUM_GROWTH_EVERY == 0:
            self.curriculum_radius = min(
                self.max_curriculum_radius,
                self.curriculum_radius + self.CURRICULUM_STEP
            )

        observation = self._get_obs()
        self.prev_dist = self._last_dist
        self.prev_abs_bearing = self._last_abs_bearing
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
    
        speed = float(action[0])
        angle = float(action[1])
        self.emitter.send(f"{speed},{angle}".encode('utf-8'))
    
        self.supervisor.step(self.timestep)
    
        obs = self._get_obs()  # this also updates self._last_dist and self._last_abs_bearing
    
        dist_to_goal = self._last_dist
        abs_bearing = self._last_abs_bearing
    
        # --- Reward shaping ---
        dist_shaping = (self.prev_dist - dist_to_goal) * 5.0
        bearing_shaping = (self.prev_abs_bearing - abs_bearing) * 2.0
    
        reward = -0.01 + dist_shaping + bearing_shaping
    
        # roll forward for next step's comparison
        self.prev_dist = dist_to_goal
        self.prev_abs_bearing = abs_bearing
    
        terminated = False
        truncated = False
    
        # Only penalize true idling: low speed AND not actively steering
        if abs(speed) < self.MIN_SPEED_THRESHOLD and abs(angle) < 0.05:
            reward += self.IDLE_PENALTY
    
        car_pos = self.vehicle_translation.getSFVec3f()
        car_x, car_y = car_pos[0], car_pos[1]
    
        if abs(car_x) > self.ARENA_LIMIT or abs(car_y) > self.ARENA_LIMIT:
            reward = -1.0
            terminated = True
            print(f"[Episode End] Hit Wall! Step: {self.current_step}", flush=True)
        elif dist_to_goal < self.COLLECT_RADIUS:
            reward = 10.0
            terminated = True
            print(f"[Episode End] Goal Reached! Step: {self.current_step}", flush=True)
        elif self.current_step >= self.MAX_STEPS:
            reward = -1.0
            truncated = True
            print(f"[Episode End] Timeout! Max actions reached.", flush=True)
    
        info = {}
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    print("Initializing Webots Gym Environment...", flush=True)
    env = WebotsGoalEnv()

    print("Starting PPO Agent Training...", flush=True)
    model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    ent_coef=0.01, 
    policy_kwargs=dict(log_std_init=0.0), 
    learning_rate=0.0003, 
    tensorboard_log="./ppo_logs/")

    model.learn(total_timesteps=50000)
    model.save("ppo_altino_model")
    print("Training Complete! Model saved.", flush=True)