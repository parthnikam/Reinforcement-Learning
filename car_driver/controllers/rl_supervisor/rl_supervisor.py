import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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

        if self.vehicle is None:
            raise RuntimeError("CRITICAL: 'VEHICLE' node not found in the Scene Tree. Ensure its DEF name is set to 'VEHICLE'.")
        if self.goal is None:
            raise RuntimeError("CRITICAL: 'GOAL' node not found in the Scene Tree. Ensure its DEF name is set to 'GOAL'.")

        # Field pointers for resetting
        self.vehicle_translation = self.vehicle.getField("translation")
        self.vehicle_rotation = self.vehicle.getField("rotation")
        self.goal_translation = self.goal.getField("translation")

        # Capture original vehicle properties to ensure clean resets
        self.initial_vehicle_pos = self.vehicle_translation.getSFVec3f()
        self.initial_vehicle_rot = self.vehicle_rotation.getSFRotation()

        # Initialize Emitter device on Supervisor
        self.emitter = self.supervisor.getDevice("emitter")
        if self.emitter is None:
            raise RuntimeError("CRITICAL: 'emitter' device not found on supervisor. Please add an Emitter base node to rl_supervisor's children.")

        # --- ENVIRONMENT CONFIGURATION ---
        self.ARENA_LIMIT = 0.9
        self.COLLECT_RADIUS = 0.15
        self.MAX_STEPS = 1200
        self.current_step = 0

        # --- PENALTY CONFIGURATION ---
        self.MIN_SPEED_THRESHOLD = 0.15
        self.IDLE_PENALTY = -0.1
        self.MIN_STEER_FOR_ACTIVE = 0.05  # steering above this counts as "actively turning", not idling

        # --- SHAPING CONFIGURATION ---
        self.HEADING_WEIGHT = 0.3   # converts radians of bearing error into "distance units" for the potential
        self.SHAPING_SCALE = 5.0
        self.JERK_PENALTY_SCALE = 0.05

        # --- CURRICULUM CONFIGURATION ---
        self.MIN_GOAL_DIST = 0.3
        self.START_GOAL_DIST = self.ARENA_LIMIT - 0.1
        self.episodes_done = 0
        self.ANGLE_CURRICULUM_EPISODES = 300
        self.DISTANCE_CURRICULUM_EPISODES = 300
        self.BEHIND_GOAL_PROB = 0.25

        # Rolling state used by shaping
        self.prev_potential = None
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        # Action Space: [Speed, Angle]
        self.action_space = spaces.Box(
            low=np.array([-1.8, -0.4], dtype=np.float32),
            high=np.array([1.8, 0.4], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space (egocentric): [dx, dy, dist, sin(yaw), cos(yaw), sin(bearing), cos(bearing)]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_heading(self):
        # Webots rotation field is axis-angle [x, y, z, angle].
        # For a vehicle rotating about the Z axis, this angle IS the yaw.
        # If your PROTO uses a different up-axis, adjust this accordingly.
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
        dist = math.sqrt(dx ** 2 + dy ** 2)

        bearing_to_goal = math.atan2(dy, dx) - yaw
        # normalize to [-pi, pi] so abs() and shaping don't break at the wraparound
        bearing_to_goal = math.atan2(math.sin(bearing_to_goal), math.cos(bearing_to_goal))

        self._last_dist = dist
        self._last_abs_bearing = abs(bearing_to_goal)

        return np.array([
            dx, dy, dist,
            math.sin(yaw), math.cos(yaw),
            math.sin(bearing_to_goal), math.cos(bearing_to_goal)
        ], dtype=np.float32)

    def _potential(self, dist, abs_bearing):
        # Lower is better. Blends distance and heading error into one scalar so
        # "back up then turn to align" isn't unconditionally penalized just
        # because distance increases for a step or two.
        return dist + self.HEADING_WEIGHT * abs_bearing

    def _spawn_curriculum_goal(self):
        angle_progress = min(1.0, self.episodes_done / self.ANGLE_CURRICULUM_EPISODES)
        distance_progress = min(1.0, self.episodes_done / self.DISTANCE_CURRICULUM_EPISODES)

        max_angle = angle_progress * math.pi / 2
        radius = self.START_GOAL_DIST - distance_progress * (self.START_GOAL_DIST - self.MIN_GOAL_DIST)
        radius = random.uniform(radius, self.START_GOAL_DIST)

        if angle_progress >= 1.0 and distance_progress >= 1.0 and random.random() < self.BEHIND_GOAL_PROB:
            theta = random.uniform(math.pi / 2, 3 * math.pi / 2)
        else:
            theta = random.uniform(-max_angle, max_angle)

        new_x = radius * math.cos(theta)
        new_y = radius * math.sin(theta)
        new_x = max(-self.ARENA_LIMIT + 0.1, min(self.ARENA_LIMIT - 0.1, new_x))
        new_y = max(-self.ARENA_LIMIT + 0.1, min(self.ARENA_LIMIT - 0.1, new_y))

        current_goal_pos = self.goal_translation.getSFVec3f()
        self.goal_translation.setSFVec3f([new_x, new_y, current_goal_pos[2]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Stop the vehicle BEFORE moving it, so no stale velocity/command carries over
        self.emitter.send("0.0,0.0".encode('utf-8'))
        self.supervisor.step(self.timestep)

        # 2. Reset transform and zero all physics (linear & angular velocity)
        self.vehicle_translation.setSFVec3f([0.0, 0.0, self.initial_vehicle_pos[2]])
        self.vehicle_rotation.setSFRotation(self.initial_vehicle_rot)
        self.vehicle.resetPhysics()

        # 3. Respawn goal from easy forward targets toward tighter turns and behind targets
        self._spawn_curriculum_goal()

        # 4. Let a few physics steps settle before the new episode's first real action
        for _ in range(3):
            self.emitter.send("0.0,0.0".encode('utf-8'))
            self.supervisor.step(self.timestep)

        self.current_step = 0
        self.episodes_done += 1

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        observation = self._get_obs()
        self.prev_potential = self._potential(self._last_dist, self._last_abs_bearing)

        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1

        speed = max(0.0, float(action[0]))
        angle = float(action[1])
        self.emitter.send(f"{speed},{angle}".encode('utf-8'))

        self.supervisor.step(self.timestep)

        obs = self._get_obs()  # updates self._last_dist and self._last_abs_bearing
        dist_to_goal = self._last_dist
        abs_bearing = self._last_abs_bearing

        # --- Potential-based shaping (distance + heading combined) ---
        current_potential = self._potential(dist_to_goal, abs_bearing)
        shaping = (self.prev_potential - current_potential) * self.SHAPING_SCALE
        self.prev_potential = current_potential

        # --- Smoothness penalty: discourage jerky, feeble, oscillating actions ---
        action_arr = np.array([speed, angle], dtype=np.float32)
        jerk_penalty = -self.JERK_PENALTY_SCALE * np.sum((action_arr - self.prev_action) ** 2)
        self.prev_action = action_arr

        reward = -0.01 + shaping + jerk_penalty
        terminated = False
        truncated = False

        # Only penalize true idling: low speed AND not actively steering
        if abs(speed) < self.MIN_SPEED_THRESHOLD and abs(angle) < self.MIN_STEER_FOR_ACTIVE:
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


# --- Training Loop Execution ---
if __name__ == "__main__":
    print("Initializing Webots Gym Environment...", flush=True)
    env = WebotsGoalEnv()
    env = Monitor(env)  # logs episode length/reward -> ep_len_mean, ep_rew_mean in console/TensorBoard

    print("Starting PPO Agent Training...", flush=True)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        ent_coef=0.01,                       # keep exploration alive longer, avoid premature steering collapse
        policy_kwargs=dict(log_std_init=0.0),  # wider initial action-noise std, especially helps the steering dim
        tensorboard_log="./ppo_logs/",
    )

    model.learn(total_timesteps=1000000)
    model.save("ppo_altino_model")
    print("Training Complete! Model saved.", flush=True)
