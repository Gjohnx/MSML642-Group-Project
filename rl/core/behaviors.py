"""Rule-based behaviors that override the Q-policy (ROS-free).

None of these are Q-learning actions. The env runs them, then gives control
back to the agent.
"""

import math

from rl.core.agent import FORWARD, TURN_LEFT, TURN_RIGHT
from rl.core.geometry import sector_clearance
from rl.core.state import FRONT, LEFT, RIGHT

# robot geometry, wheel separation 0.29 m + 0.04 m tires (see sim/room.sdf)
ROBOT_WIDTH_M = 0.33
CLEARANCE_MARGIN_M = 0.12

# collision recovery
BACKUP_SPEED = 0.10
TURN_SPEED = 0.6
MAX_BACKUP_M = 0.20
MIN_BACKUP_M = 0.05
REAR_CLEAR_M = 0.25
REAR_MARGIN_M = 0.12
RECOVERY_TURN_DEG = 35.0
RECOVERY_STOP_S = 0.25

# lost-target search
SCAN_STEPS = 20
EXPLORE_FRONT_M = 1.0


# raw velocity command the env publishes on /robot/cmd_vel for duration_s of sim time
class Command:
    def __init__(self, linear, angular, duration_s):
        self.linear = linear
        self.angular = angular
        self.duration_s = duration_s


# stop distances derived from the robot width and a clearance margin
class SafetyConfig:
    def __init__(
        self, robot_width_m=ROBOT_WIDTH_M, clearance_margin=CLEARANCE_MARGIN_M
    ):
        half_width = robot_width_m / 2.0
        self.robot_width_m = robot_width_m
        self.clearance_margin = clearance_margin
        self.front_stop = 0.10 + clearance_margin
        self.side_stop = half_width + 0.3 * clearance_margin


# checks whether moving toward a direction (robot frame, radians) gets too close
# rays are (angle, distance, hit) tuples from geometry.scan_to_rays()
def motion_unsafe(rays, range_max, direction, safety):
    hit_rays = []
    for ray in rays:
        if ray[2]:
            hit_rays.append(ray)
    half_pi = math.pi / 2.0
    front = sector_clearance(hit_rays, direction, math.radians(25.0), range_max)
    center = sector_clearance(hit_rays, direction, math.radians(8.0), range_max)
    left = sector_clearance(
        hit_rays, direction + half_pi, math.radians(30.0), range_max
    )
    right = sector_clearance(
        hit_rays, direction - half_pi, math.radians(30.0), range_max
    )
    if front < safety.front_stop or center < safety.front_stop:
        return True
    return left < safety.side_stop or right < safety.side_stop


# reads a clearance and treats a missing reading as no room at all
def clearance_or_zero(clear, name):
    value = clear.get(name)
    if not value:
        return 0.0
    return value


# stop, back up if the rear is clear, then turn toward the open side
# backs up min(0.20, rear - 0.12) when the rear clearance is above 0.25 m
# a rear sector with no lidar coverage is unsafe, so the robot turns in place
# rear_offset_m is the distance from the lidar back to the rear of the body
class CollisionRecovery:
    def __init__(
        self,
        backup_speed=BACKUP_SPEED,
        turn_speed=TURN_SPEED,
        max_backup_m=MAX_BACKUP_M,
        rear_clear_m=REAR_CLEAR_M,
        rear_margin_m=REAR_MARGIN_M,
        min_backup_m=MIN_BACKUP_M,
        turn_deg=RECOVERY_TURN_DEG,
        stop_s=RECOVERY_STOP_S,
        rear_offset_m=0.0,
    ):
        self.backup_speed = backup_speed
        self.turn_speed = turn_speed
        self.max_backup_m = max_backup_m
        self.rear_clear_m = rear_clear_m
        self.rear_margin_m = rear_margin_m
        self.min_backup_m = min_backup_m
        self.turn_rad = math.radians(turn_deg)
        self.stop_s = stop_s
        self.rear_offset_m = rear_offset_m

    # clear is geometry.clearances(): keys rear/left/right, values in meters or None
    def plan(self, clear):
        commands = []
        commands.append(Command(0.0, 0.0, self.stop_s))

        rear = clear.get("rear")
        if rear is not None:
            rear = rear - self.rear_offset_m
        backup = 0.0
        if rear is not None and rear > self.rear_clear_m:
            backup = min(self.max_backup_m, rear - self.rear_margin_m)
        if backup >= self.min_backup_m:
            duration = backup / self.backup_speed
            commands.append(Command(-self.backup_speed, 0.0, duration))

        left = clearance_or_zero(clear, "left")
        right = clearance_or_zero(clear, "right")
        direction = -1.0
        if left >= right:
            direction = 1.0
        duration = self.turn_rad / self.turn_speed
        commands.append(Command(0.0, direction * self.turn_speed, duration))
        return commands


# counts separate collisions, a new one only after contact has cleared
# the episode ends once max_collisions is reached
class CollisionCounter:
    def __init__(self, max_collisions=3):
        self.max_collisions = max_collisions
        self.reset()

    def reset(self):
        self.count = 0
        self.in_contact = False
        self.done = False

    # takes this step's contact flag and returns True for a new collision
    def observe(self, in_contact):
        new_collision = in_contact and not self.in_contact
        self.in_contact = in_contact
        if new_collision:
            self.count += 1
        self.done = self.count >= self.max_collisions
        return new_collision


# runs while the NPC is out of view, the env hands control back once it is seen
# scan phase: turn toward the last-seen side for up to scan_steps steps
# explore phase: drive forward while the front is open, else turn to the open side
class LostTargetSearch:
    def __init__(self, scan_steps=SCAN_STEPS, explore_front_m=EXPLORE_FRONT_M):
        self.scan_steps = scan_steps
        self.explore_front_m = explore_front_m
        self.reset()

    def reset(self):
        self.steps = 0

    # returns a robot action so the search uses the same step timing as the agent
    def next_action(self, last_seen_dir, clear):
        self.steps += 1
        if self.steps <= self.scan_steps:
            if last_seen_dir == RIGHT:
                return TURN_RIGHT
            if last_seen_dir == LEFT:
                return TURN_LEFT
            front = clearance_or_zero(clear, "front")
            if last_seen_dir == FRONT and front > self.explore_front_m:
                # it was ahead, keep going the way it went
                return FORWARD
        return self.explore(clear)

    def explore(self, clear):
        front = clearance_or_zero(clear, "front")
        if front > self.explore_front_m:
            return FORWARD
        left = clearance_or_zero(clear, "left")
        right = clearance_or_zero(clear, "right")
        if left >= right:
            return TURN_LEFT
        return TURN_RIGHT
