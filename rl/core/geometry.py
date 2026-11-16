"""Pose and lidar helpers (ROS-free).

Functions take plain numbers or duck-typed messages (anything with the same
attributes as the ROS message), so they can be unit-tested without ROS.
"""

import math

# lidar sectors: name -> (center, half width) in radians
SECTORS = {
    "front": (0.0, math.radians(25.0)),
    "center_front": (0.0, math.radians(8.0)),
    "left": (math.pi / 2.0, math.radians(30.0)),
    "right": (-math.pi / 2.0, math.radians(30.0)),
    "rear": (math.pi, math.radians(30.0)),
}


# wraps an angle to [-pi, pi]
def normalization(a):
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    return math.atan2(sin_a, cos_a)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_from_odom(odom):
    q = odom.pose.pose.orientation
    return yaw_from_quaternion(q)


# returns (x, y, yaw) from an odometry message
def pose_from_odom(odom):
    position = odom.pose.pose.position
    yaw = yaw_from_odom(odom)
    return (position.x, position.y, yaw)


# calculates a target point at a distance and heading offset from the robot
def relative_waypoint(pose, distance, heading_offset=0.0):
    px, py, yaw = pose
    heading = normalization(yaw + heading_offset)
    target_x = px + distance * math.cos(heading)
    target_y = py + distance * math.sin(heading)
    return (target_x, target_y, heading)


def robot_to_world(pose, point):
    px, py, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = px + point[0] * cos_yaw - point[1] * sin_yaw
    world_y = py + point[0] * sin_yaw + point[1] * cos_yaw
    return (world_x, world_y)


# returns the bearing and distance of a world point as seen from the robot
def world_to_robot(pose, point):
    px, py, yaw = pose
    dx = point[0] - px
    dy = point[1] - py
    bearing = normalization(math.atan2(dy, dx) - yaw)
    distance = math.hypot(dx, dy)
    return bearing, distance


# turns a LaserScan into (angle, distance, hit) rays
# a miss (inf, NaN, out of range) gets distance = range_max and hit = False
def scan_to_rays(scan):
    range_max = float(scan.range_max)
    rays = []
    for i in range(len(scan.ranges)):
        raw = scan.ranges[i]
        angle = scan.angle_min + i * scan.angle_increment
        hit = False
        if math.isfinite(raw):
            above_min = raw >= scan.range_min
            below_max = raw <= scan.range_max
            hit = above_min and below_max
        distance = range_max
        if hit:
            distance = float(raw)
        rays.append((angle, distance, hit))
    return rays


# returns minimum lidar range between angles
def sector_clearance(rays, center, half_width, default):
    smallest = None
    for angle, distance, hit in rays:
        offset = normalization(angle - center)
        if abs(offset) <= half_width:
            if smallest is None or distance < smallest:
                smallest = distance
    if smallest is None:
        return default
    if smallest > default:
        return default
    return smallest


# checks whether at least one beam points into the sector
# a sector outside the lidar's field of view must not be mistaken for clear
def sector_covered(rays, center, half_width):
    for angle, distance, hit in rays:
        offset = normalization(angle - center)
        if abs(offset) <= half_width:
            return True
    return False


# returns the clearance of every sector in meters, None for a sector with no beams
def clearances(rays, range_max):
    hit_rays = []
    for ray in rays:
        if ray[2]:
            hit_rays.append(ray)
    result = {}
    for name in SECTORS:
        center, half_width = SECTORS[name]
        if sector_covered(rays, center, half_width):
            result[name] = sector_clearance(hit_rays, center, half_width, range_max)
        else:
            result[name] = None
    return result
