"""Limited-range, limited-FOV NPC detector (ROS-free math; the env feeds it poses).

Uses ground-truth learner and NPC poses, but only reports the NPC when it is
within range_m, inside fov_deg, and not hidden behind an obstacle: the lidar
beam at the NPC's bearing must not hit something clearly closer than the NPC.
Returns (visible, bearing_rad, distance_m) in the learner's frame.
"""

import math

from rl.core.geometry import normalization, world_to_robot

# a hit this much closer than the NPC means something is in the way
OCCLUSION_MARGIN_M = 0.3
# the lidar sits 0.12 m ahead of the base origin (sim/room.sdf)
LIDAR_OFFSET_M = 0.12


# returns the lidar ray that points closest to a bearing
def nearest_ray(rays, bearing):
    best = None
    best_offset = None
    for ray in rays:
        offset = abs(normalization(ray[0] - bearing))
        if best_offset is None or offset < best_offset:
            best = ray
            best_offset = offset
    return best


# learner_pose is (x, y, yaw) of the robot base, rays come from scan_to_rays()
# ranges are compared from the lidar position, not the base origin
def detect(
    learner_pose,
    npc_xy,
    rays,
    range_m,
    fov_deg,
    occlusion_margin_m=OCCLUSION_MARGIN_M,
    lidar_offset_m=LIDAR_OFFSET_M,
):
    if not rays:
        return False, None, None

    bearing, distance = world_to_robot(learner_pose, npc_xy)
    half_fov = math.radians(fov_deg) / 2.0
    if distance > range_m or abs(bearing) > half_fov:
        return False, None, None

    x, y, yaw = learner_pose
    lidar_x = x + lidar_offset_m * math.cos(yaw)
    lidar_y = y + lidar_offset_m * math.sin(yaw)
    lidar_pose = (lidar_x, lidar_y, yaw)
    lidar_bearing, lidar_distance = world_to_robot(lidar_pose, npc_xy)

    ray = nearest_ray(rays, lidar_bearing)
    ray_distance = ray[1]
    ray_hit = ray[2]
    if ray_hit and ray_distance < lidar_distance - occlusion_margin_m:
        return False, None, None
    return True, bearing, distance
