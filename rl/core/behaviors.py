"""Rule-based behaviors that override the Q-policy (ROS-free).

None of these are Q-learning actions. The env runs them, then gives control
back to the agent.
"""

import math
from dataclasses import dataclass

from rl.core.agent import FORWARD, TURN_LEFT, TURN_RIGHT
from rl.core.geometry import sector_clearance
from rl.core.state import FRONT, LEFT, RIGHT


@dataclass(frozen=True)
class Command:
    """Raw velocity command the env publishes on /robot/cmd_vel for duration_s of sim time."""
    linear: float
    angular: float
    duration_s: float


# ---------------------------------------------------------------- safety check

@dataclass(frozen=True)
class SafetyConfig:
    """Stop distances derived from robot width and a clearance margin.
    The robot is ~0.33 m wide (wheel separation 0.29 m + 0.04 m tires, see sim/room.sdf)."""
    robot_width_m: float = 0.33
    clearance_margin: float = 0.12

    @property
    def front_stop(self) -> float:
        return 0.10 + self.clearance_margin

    @property
    def side_stop(self) -> float:
        return self.robot_width_m / 2.0 + 0.3 * self.clearance_margin


def motion_unsafe(rays, range_max, direction, safety: SafetyConfig) -> bool:
    """True if moving toward `direction` (robot frame, rad) would get too close.
    `rays` are (angle, distance, hit) tuples from geometry.scan_to_rays()."""
    hits = [r for r in rays if r[2]]
    half_pi = math.pi / 2.0
    front = sector_clearance(hits, direction, math.radians(25.0), range_max)
    center = sector_clearance(hits, direction, math.radians(8.0), range_max)
    left = sector_clearance(hits, direction + half_pi, math.radians(30.0), range_max)
    right = sector_clearance(hits, direction - half_pi, math.radians(30.0), range_max)
    if front < safety.front_stop or center < safety.front_stop:
        return True
    return left < safety.side_stop or right < safety.side_stop


# ---------------------------------------------------------- collision recovery

class CollisionRecovery:
    """stop -> back up (<= 0.2 m) if the rear is clear -> turn toward the open side.

    Backs up min(0.20, rear - 0.12) when the rear clearance is above 0.25 m and
    the backup is at least 0.05 m, then turns 35 deg toward whichever side has
    more room. A rear sector with no lidar coverage (None) is treated as
    unsafe, so the robot turns in place instead.
    """

    def __init__(self, backup_speed=0.10, turn_speed=0.6, max_backup_m=0.20,
                 rear_clear_m=0.25, rear_margin_m=0.12, min_backup_m=0.05,
                 turn_deg=35.0, stop_s=0.25):
        self.backup_speed = backup_speed
        self.turn_speed = turn_speed
        self.max_backup_m = max_backup_m
        self.rear_clear_m = rear_clear_m
        self.rear_margin_m = rear_margin_m
        self.min_backup_m = min_backup_m
        self.turn_rad = math.radians(turn_deg)
        self.stop_s = stop_s

    def plan(self, clear: dict) -> list[Command]:
        """`clear` is geometry.clearances(): keys rear/left/right, values in m or None."""
        commands = [Command(0.0, 0.0, self.stop_s)]
        rear = clear.get("rear")
        backup = 0.0
        if rear is not None and rear > self.rear_clear_m:
            backup = min(self.max_backup_m, rear - self.rear_margin_m)
        if backup >= self.min_backup_m:
            commands.append(Command(-self.backup_speed, 0.0, backup / self.backup_speed))
        left = clear.get("left") or 0.0
        right = clear.get("right") or 0.0
        direction = 1.0 if left >= right else -1.0
        commands.append(Command(0.0, direction * self.turn_speed, self.turn_rad / self.turn_speed))
        return commands


class CollisionCounter:
    """Counts separate collisions (a new one only after contact has cleared).
    The episode ends once max_collisions is reached."""

    def __init__(self, max_collisions: int = 3):
        self.max_collisions = max_collisions
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self._in_contact = False

    def observe(self, in_contact: bool) -> bool:
        """Feed this step's contact flag. Returns True if this is a new collision."""
        new = in_contact and not self._in_contact
        self._in_contact = in_contact
        if new:
            self.count += 1
        return new

    @property
    def done(self) -> bool:
        return self.count >= self.max_collisions


# ---------------------------------------------------------- lost-target search

class LostTargetSearch:
    """Runs while the NPC is out of view; the env hands control back once it is seen.

    Phase 1 (scan): turn toward the last-seen side for up to scan_steps steps.
    Phase 2 (explore): drive forward while the front is open, otherwise turn
    toward the side with more room.
    Returns primitive action indices so it uses the same step timing as the agent.
    """

    def __init__(self, scan_steps: int = 20, explore_front_m: float = 1.0):
        self.scan_steps = scan_steps
        self.explore_front_m = explore_front_m
        self.reset()

    def reset(self) -> None:
        self.steps = 0

    def next_action(self, last_seen_dir: int, clear: dict) -> int:
        self.steps += 1
        if self.steps <= self.scan_steps:
            if last_seen_dir == RIGHT:
                return TURN_RIGHT
            if last_seen_dir == LEFT:
                return TURN_LEFT
            if last_seen_dir == FRONT and (clear.get("front") or 0.0) > self.explore_front_m:
                return FORWARD     # it was ahead: keep going the way it went
        return self._explore(clear)

    def _explore(self, clear: dict) -> int:
        if (clear.get("front") or 0.0) > self.explore_front_m:
            return FORWARD
        left = clear.get("left") or 0.0
        right = clear.get("right") or 0.0
        return TURN_LEFT if left >= right else TURN_RIGHT
