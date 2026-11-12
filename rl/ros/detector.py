"""Limited-range, limited-FOV NPC detector (ROS-free math; the env feeds it poses).

Uses ground-truth learner and NPC poses, but only reports the NPC when it is
within range_m, inside fov_deg, and not hidden behind an obstacle: the lidar
beam at the NPC's bearing must not hit something clearly closer than the NPC.
Returns (visible, bearing_rad, distance_m) in the learner's frame.
"""

import math

from rl.core.geometry import normalization, world_to_robot


def detect(learner_pose, npc_xy, rays, range_m: float, fov_deg: float,
           occlusion_margin_m: float = 0.3, lidar_offset_m: float = 0.12):
    """learner_pose = (x, y, yaw) of the robot base; rays from scan_to_rays().

    lidar_offset_m: the lidar sits 0.12 m ahead of the base origin (sim/room.sdf),
    so ranges are compared from the lidar position.
    """
    bearing, distance = world_to_robot(learner_pose, npc_xy)
    if distance > range_m or abs(bearing) > math.radians(fov_deg) / 2.0:
        return False, None, None
    x, y, yaw = learner_pose
    lidar_pose = (x + lidar_offset_m * math.cos(yaw), y + lidar_offset_m * math.sin(yaw), yaw)
    lidar_bearing, lidar_distance = world_to_robot(lidar_pose, npc_xy)
    if rays:
        beam = min(rays, key=lambda r: abs(normalization(r[0] - lidar_bearing)))
        if beam[2] and beam[1] < lidar_distance - occlusion_margin_m:
            return False, None, None
    return True, bearing, distance
