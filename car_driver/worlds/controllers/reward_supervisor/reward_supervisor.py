from controller import Supervisor
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

vehicle = supervisor.getFromDef("VEHICLE")
reward_1 = supervisor.getFromDef("REWARD_1")
reward_2 = supervisor.getFromDef("REWARD_2")
goal = supervisor.getFromDef("GOAL")

score = 0
collected_rewards = set()

COLLECT_RADIUS = 0.18
GOAL_RADIUS = 0.25


def distance_2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def get_position(node):
    return node.getField("translation").getSFVec3f()


def hide_node(node):
    translation_field = node.getField("translation")
    pos = translation_field.getSFVec3f()
    translation_field.setSFVec3f([pos[0], pos[1], -10.0])


while supervisor.step(timestep) != -1:
    vehicle_pos = get_position(vehicle)

    rewards = [
        ("REWARD_1", reward_1),
        ("REWARD_2", reward_2),
    ]

    for reward_name, reward_node in rewards:
        if reward_name in collected_rewards:
            continue

        reward_pos = get_position(reward_node)

        if distance_2d(vehicle_pos, reward_pos) < COLLECT_RADIUS:
            collected_rewards.add(reward_name)
            score += 1
            hide_node(reward_node)

            print(f"Collected {reward_name}. Score = {score}")

    goal_pos = get_position(goal)

    if distance_2d(vehicle_pos, goal_pos) < GOAL_RADIUS:
        print(f"Goal reached. Final score = {score}")
        supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)