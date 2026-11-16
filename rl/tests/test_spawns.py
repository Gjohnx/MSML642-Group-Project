"""Check configured start poses and NPC routes against the obstacles in sim/room.sdf.

Pure geometry on the world file: no Gazebo needed. Run from the repo root.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from rl.core.geometry import world_to_robot

CFG = yaml.safe_load(Path("rl/config/qlearning.yaml").read_text())
ROUTES = yaml.safe_load(Path("sim/npc_routes.yaml").read_text())["routes"]

# map limits, the room is 16 x 12 m centered on the origin
MAP_HALF_X = 7.8
MAP_HALF_Y = 5.8

# clearance limits
ROBOT_CLEARANCE_M = 0.35
ROUTE_CLEARANCE_M = 0.30
MIN_OBSTACLE_TOP_M = 0.10
SIGHT_STEP_M = 0.05
ROUTE_STEP_M = 0.10


# reads the static collision shapes that can block the robot from the world file
def load_obstacles():
    shapes = []
    world = ET.parse("sim/room.sdf").getroot().find("world")
    for model in world.findall("model"):
        if model.findtext("static") != "true":
            continue
        pose = model.findtext("pose", "0 0 0 0 0 0").split()
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2])
        yaw = float(pose[5])
        for collision in model.iter("collision"):
            box = collision.find("geometry/box/size")
            cylinder = collision.find("geometry/cylinder")
            if box is not None:
                size = box.text.split()
                top = z + float(size[2]) / 2.0
                # skip the floor and the floor markings
                if top > MIN_OBSTACLE_TOP_M:
                    half_x = float(size[0]) / 2.0
                    half_y = float(size[1]) / 2.0
                    shapes.append(("box", x, y, yaw, half_x, half_y))
            elif cylinder is not None:
                radius = float(cylinder.findtext("radius"))
                shapes.append(("cylinder", x, y, 0.0, radius, 0.0))
    return shapes


SHAPES = load_obstacles()


# distance from a point to the nearest obstacle surface, negative when inside one
def clearance(x, y):
    best = math.inf
    for kind, shape_x, shape_y, yaw, a, b in SHAPES:
        dx = x - shape_x
        dy = y - shape_y
        if kind == "cylinder":
            distance = math.hypot(dx, dy) - a
        else:
            local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            outside_x = abs(local_x) - a
            outside_y = abs(local_y) - b
            outside = math.hypot(max(outside_x, 0.0), max(outside_y, 0.0))
            inside = min(max(outside_x, outside_y), 0.0)
            distance = outside + inside
        if distance < best:
            best = distance
    return best


# returns evenly spaced points from start to end
def points_between(start, end, step):
    count = int(math.dist(start, end) / step)
    if count < 1:
        count = 1
    points = []
    for i in range(count + 1):
        t = i / count
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        points.append((x, y))
    return points


# checks that no obstacle sits on the straight line between two points
def line_of_sight(start, end):
    for x, y in points_between(start, end, SIGHT_STEP_M):
        if clearance(x, y) <= 0:
            return False
    return True


# returns (set name, pose) for the default starts and every named start set
def all_spawns():
    spawns = []
    for spawn in CFG["spawns"]:
        spawns.append(("default", spawn))
    for name in CFG["spawn_sets"]:
        for spawn in CFG["spawn_sets"][name]["spawns"]:
            spawns.append((name, spawn))
    return spawns


def test_world_file_was_parsed():
    assert len(SHAPES) > 20
    # the corridor center is open
    assert clearance(0.0, 0.0) > 0.5
    # this point is inside the aisle barrier
    assert clearance(2.0, 0.0) < 0


def test_every_spawn_is_inside_the_map_and_clear_of_obstacles():
    for name, spawn in all_spawns():
        x = spawn[0]
        y = spawn[1]
        assert abs(x) < MAP_HALF_X and abs(y) < MAP_HALF_Y, (name, x, y)
        gap = clearance(x, y)
        assert gap >= ROBOT_CLEARANCE_M, (name, x, y, round(gap, 2))


def test_tracking_spawns_see_the_npc_at_reset():
    group = CFG["spawn_sets"]["tracking"]
    max_range = CFG["detector"]["range_m"]
    half_fov = math.radians(CFG["detector"]["fov_deg"]) / 2.0
    min_range = CFG["capture_distance_m"] + 0.5
    for route in group["routes"]:
        npc = ROUTES[route]["waypoints"][0]
        for spawn in group["spawns"]:
            pose = (spawn[0], spawn[1], spawn[2])
            bearing, distance = world_to_robot(pose, npc)
            assert min_range < distance <= max_range, (spawn, distance)
            assert abs(bearing) <= half_fov, (spawn, bearing)
            assert line_of_sight((spawn[0], spawn[1]), npc), spawn


def test_heldout_spawns_are_not_training_spawns():
    training = []
    for spawn in CFG["spawns"]:
        training.append((spawn[0], spawn[1]))
    for spawn in CFG["spawn_sets"]["tracking"]["spawns"]:
        training.append((spawn[0], spawn[1]))
    for spawn in CFG["spawn_sets"]["heldout"]["spawns"]:
        nearest = math.inf
        for point in training:
            distance = math.dist((spawn[0], spawn[1]), point)
            if distance < nearest:
                nearest = distance
        assert nearest > 1.0, spawn


def test_npc_routes_stay_clear_of_obstacles():
    for name in ROUTES:
        waypoints = ROUTES[name]["waypoints"]
        for i in range(len(waypoints)):
            start = waypoints[i]
            end = waypoints[(i + 1) % len(waypoints)]
            for x, y in points_between(start, end, ROUTE_STEP_M):
                gap = clearance(x, y)
                assert gap >= ROUTE_CLEARANCE_M, (name, round(x, 2), round(y, 2))
