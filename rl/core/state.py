"""Discrete state encoding (ROS-free).

State tuple (all ints):
    npc_visible    : 0 / 1
    npc_direction  : LEFT / FRONT / RIGHT, or NONE when not visible
    npc_distance   : NEAR / MEDIUM / FAR, or NONE when not visible
    last_seen_dir  : LEFT / FRONT / RIGHT, or NONE if never seen this episode
    obstacle_front : 0 = near .. k-1 = far   (bins from cfg["obstacle_bins_m"])
    obstacle_left  : same
    obstacle_right : same

encode() maps the tuple to one Q-table row. Invisible states collapse the NPC
fields to a single value, so with 3 obstacle levels there are
(1 + 3*3) * 3**3 * 4 = 1080 states.

The obstacle inputs come from geometry.clearances().
"""

import math

# directions
LEFT = 0
FRONT = 1
RIGHT = 2
NONE = 3

# distance and obstacle levels
NEAR = 0
MEDIUM = 1
FAR = 2

# NPC bearing within +-15 deg counts as FRONT
DEFAULT_FRONT_HALF_DEG = 15.0


# one observation in the robot frame, produced by rl.ros
# ranges are the minimum over each lidar sector from geometry.clearances()
class Observation:
    def __init__(
        self,
        npc_visible,
        npc_bearing_rad,
        npc_distance_m,
        range_front_m,
        range_left_m,
        range_right_m,
    ):
        self.npc_visible = npc_visible
        self.npc_bearing_rad = npc_bearing_rad
        self.npc_distance_m = npc_distance_m
        self.range_front_m = range_front_m
        self.range_left_m = range_left_m
        self.range_right_m = range_right_m


# returns the index of the first edge the value is below
def bin_value(value, edges):
    for i in range(len(edges)):
        if value < edges[i]:
            return i
    return len(edges)


# positive yaw is counter-clockwise, so a positive bearing is to the left
def bearing_to_direction(bearing_rad, front_half_deg=DEFAULT_FRONT_HALF_DEG):
    half = math.radians(front_half_deg)
    if bearing_rad > half:
        return LEFT
    if bearing_rad < -half:
        return RIGHT
    return FRONT


# turns an observation into the discrete state tuple
def discretize(obs, last_seen_dir, cfg):
    obstacle_edges = cfg["obstacle_bins_m"]
    direction = NONE
    distance = NONE
    if obs.npc_visible:
        front_half_deg = cfg.get("npc_front_half_deg", DEFAULT_FRONT_HALF_DEG)
        direction = bearing_to_direction(obs.npc_bearing_rad, front_half_deg)
        distance = bin_value(obs.npc_distance_m, cfg["npc_distance_bins_m"])
    visible = int(obs.npc_visible)
    front = bin_value(obs.range_front_m, obstacle_edges)
    left = bin_value(obs.range_left_m, obstacle_edges)
    right = bin_value(obs.range_right_m, obstacle_edges)
    return (visible, direction, distance, last_seen_dir, front, left, right)


# remembers the NPC direction while visible, keeps the old one while hidden
def next_last_seen(state, previous_last_seen):
    visible = state[0]
    direction = state[1]
    if visible:
        return direction
    return previous_last_seen


# returns how many obstacle levels and NPC distance levels the config defines
def level_counts(cfg):
    obstacle_levels = len(cfg["obstacle_bins_m"]) + 1
    distance_levels = len(cfg["npc_distance_bins_m"]) + 1
    return obstacle_levels, distance_levels


def num_states(cfg):
    obstacle_levels, distance_levels = level_counts(cfg)
    npc_values = 1 + 3 * distance_levels
    obstacle_values = obstacle_levels**3
    return npc_values * obstacle_values * 4


# maps a state tuple to its row in the Q-table
def encode(state, cfg):
    obstacle_levels, distance_levels = level_counts(cfg)
    visible, direction, distance, last_seen, front, left, right = state
    npc = 0
    if visible:
        npc = 1 + direction * distance_levels + distance
    index = npc
    index = index * obstacle_levels + front
    index = index * obstacle_levels + left
    index = index * obstacle_levels + right
    index = index * 4 + last_seen
    return index
