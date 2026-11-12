"""Discrete state encoding (ROS-free).

State tuple (all ints):
    npc_visible    : 0 / 1
    npc_direction  : LEFT / FRONT / RIGHT, or NONE when not visible
    npc_distance   : NEAR / MEDIUM / FAR, or NONE when not visible
    obstacle_front : 0 = near .. k-1 = far   (bins from cfg["obstacle_bins_m"])
    obstacle_left  : same
    obstacle_right : same
    last_seen_dir  : LEFT / FRONT / RIGHT, or NONE if never seen this episode

encode() maps the tuple to one Q-table row. Invisible states collapse the NPC
fields to a single value, so with 3 obstacle levels there are
(1 + 3*3) * 3**3 * 4 = 1080 states.

The obstacle inputs come from geometry.clearances().
"""

import math
from dataclasses import dataclass

LEFT, FRONT, RIGHT, NONE = 0, 1, 2, 3
NEAR, MEDIUM, FAR = 0, 1, 2

DEFAULT_FRONT_HALF_DEG = 15.0   # NPC bearing within +-15 deg counts as FRONT


@dataclass(frozen=True)
class Observation:
    """Inputs produced by rl.ros (robot frame). Ranges are the min over each
    /robot/scan sector, from geometry.clearances()."""
    npc_visible: bool
    npc_bearing_rad: float | None
    npc_distance_m: float | None
    range_front_m: float
    range_left_m: float
    range_right_m: float


def bin_value(value, edges):
    """Index of the first edge that value is below; len(edges) if above all."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def bearing_to_direction(bearing_rad, front_half_deg=DEFAULT_FRONT_HALF_DEG):
    half = math.radians(front_half_deg)
    if bearing_rad > half:
        return LEFT          # positive yaw is counter-clockwise = left
    if bearing_rad < -half:
        return RIGHT
    return FRONT


def discretize(obs: Observation, last_seen_dir: int, cfg: dict) -> tuple:
    obstacle_edges = cfg["obstacle_bins_m"]
    if obs.npc_visible:
        direction = bearing_to_direction(
            obs.npc_bearing_rad, cfg.get("npc_front_half_deg", DEFAULT_FRONT_HALF_DEG))
        distance = bin_value(obs.npc_distance_m, cfg["npc_distance_bins_m"])
    else:
        direction, distance = NONE, NONE
    return (
        int(obs.npc_visible),
        direction,
        distance,
        bin_value(obs.range_front_m, obstacle_edges),
        bin_value(obs.range_left_m, obstacle_edges),
        bin_value(obs.range_right_m, obstacle_edges),
        last_seen_dir,
    )


def next_last_seen(state: tuple, previous_last_seen: int) -> int:
    """Remember the NPC direction while visible; keep the old one while hidden."""
    return state[1] if state[0] else previous_last_seen


def _sizes(cfg: dict):
    n_obs = len(cfg["obstacle_bins_m"]) + 1
    n_dist = len(cfg["npc_distance_bins_m"]) + 1
    return n_obs, n_dist


def num_states(cfg: dict) -> int:
    n_obs, n_dist = _sizes(cfg)
    return (1 + 3 * n_dist) * n_obs ** 3 * 4


def encode(state: tuple, cfg: dict) -> int:
    n_obs, n_dist = _sizes(cfg)
    visible, direction, distance, front, left, right, last_seen = state
    npc = 0 if not visible else 1 + direction * n_dist + distance
    index = npc
    for value in (front, left, right):
        index = index * n_obs + value
    return index * 4 + last_seen
