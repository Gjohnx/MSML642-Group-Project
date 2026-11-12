"""Pose and lidar helpers (ROS-free).

Functions take plain numbers or duck-typed messages (anything with the same
attributes as the ROS message), so they can be unit-tested without ROS.
"""

import math

# Lidar sectors: name -> (center_rad, half_width_rad).
SECTORS = {
    "front": (0.0, math.radians(25.0)),
    "center_front": (0.0, math.radians(8.0)),
    "left": (math.pi / 2.0, math.radians(30.0)),
    "right": (-math.pi / 2.0, math.radians(30.0)),
    "rear": (math.pi, math.radians(30.0)),
}


def normalization(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_from_odom(odom):
    return yaw_from_quaternion(odom.pose.pose.orientation)


def pose_from_odom(odom):
    """(x, y, yaw) from a nav_msgs/Odometry."""
    p = odom.pose.pose.position
    return (p.x, p.y, yaw_from_odom(odom))


def relative_waypoint(pose, distance, heading_offset=0.0):
    px, py, yaw = pose
    heading = normalization(yaw + heading_offset)
    return (px + distance * math.cos(heading), py + distance * math.sin(heading), heading)


def robot_to_world(pose, point):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (px + point[0] * c - point[1] * s, py + point[0] * s + point[1] * c)


def world_to_robot(pose, point):
    """Inverse of robot_to_world: returns (bearing_rad, distance_m) of a world point."""
    px, py, yaw = pose
    dx, dy = point[0] - px, point[1] - py
    return normalization(math.atan2(dy, dx) - yaw), math.hypot(dx, dy)


def scan_to_rays(scan):
    """(angle, distance, hit) for every beam. Misses (inf, NaN, out of range)
    get distance=range_max, hit=False."""
    range_max = float(scan.range_max)
    rays = []
    for i, raw in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment
        hit = math.isfinite(raw) and scan.range_min <= raw <= scan.range_max
        rays.append((angle, float(raw) if hit else range_max, hit))
    return rays


def sector_clearance(rays, center, half_width, default):
    """Smallest range among rays within half_width of center, capped at default."""
    smallest = None
    for angle, distance, _hit in rays:
        if abs(normalization(angle - center)) <= half_width:
            if smallest is None or distance < smallest:
                smallest = distance
    if smallest is None or smallest > default:
        return default
    return smallest


def sector_covered(rays, center, half_width):
    """True if at least one beam points into the sector. The robot's lidar is
    180 deg, so "rear" has no beams and must not be mistaken for "clear"."""
    return any(abs(normalization(a - center)) <= half_width for a, _d, _h in rays)


def clearances(rays, range_max):
    """Per-sector clearance in meters. A sector with no beams is None."""
    hits = [r for r in rays if r[2]]
    out = {}
    for name, (center, half) in SECTORS.items():
        if not sector_covered(rays, center, half):
            out[name] = None
        else:
            out[name] = sector_clearance(hits, center, half, range_max)
    return out
