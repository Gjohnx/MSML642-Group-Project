"""Discrete state encoding (ROS-free).

State tuple:
    npc_visible      : 0/1
    npc_direction    : LEFT / FRONT / RIGHT / NONE
    npc_distance     : NEAR / MEDIUM / FAR / NONE
    obstacle_front   : 0..k-1  (binned range reading)
    obstacle_left    : 0..k-1
    obstacle_right   : 0..k-1
    last_seen_dir    : LEFT / FRONT / RIGHT / NONE

`encode()` maps the tuple to a single integer index into the Q-table.
"""

from dataclasses import dataclass

LEFT, FRONT, RIGHT, NONE = 0, 1, 2, 3


@dataclass(frozen=True)
class Observation:
    """Inputs produced by rl.ros (robot frame). Ranges are the min over each
    /robot/scan sector defined in config scan_sectors_deg."""
    npc_visible: bool
    npc_bearing_rad: float | None
    npc_distance_m: float | None
    range_front_m: float
    range_left_m: float
    range_right_m: float


def discretize(obs: Observation, last_seen_dir: int, cfg: dict) -> tuple:
    raise NotImplementedError


def encode(state: tuple) -> int:
    raise NotImplementedError


def num_states(cfg: dict) -> int:
    raise NotImplementedError
