"""ROS 2 <-> RL bridge for the sim in sim/ (the only part of rl/ that talks to ROS).

Topics (sim/bridge.yaml):
    /robot/cmd_vel  Twist      publish the chosen action for action_duration_s of sim time
    /robot/odom     Odometry   learner pose / step timing
    /robot/scan     LaserScan  181 beams, -90..+90 deg, 0.05-6.0 m
    /clock          Clock      sim time
    /robot/contact  Contacts   not in the sim yet (see docs/SIM_TODO.md)
    /npc/odom       Odometry   not in the sim yet (see docs/SIM_TODO.md)

Interface:
    reset(spawn=None, route=None) -> Observation
    step(action) -> (Observation, info)

`info` holds ground truth (learner pose, true NPC pose, region id, contact,
captured). Used for rewards and metrics only; never part of the state.

The helpers below do not import rclpy, so they can be unit-tested;
ChaseEnv imports ROS lazily.
"""

import time

from rl.core.state import Observation  # noqa: F401  (ChaseEnv returns these)

MAX_SENSOR_AGE_S = 2.0


def stamp_seconds(stamp) -> float:
    """builtin_interfaces/Time -> float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def external_contacts(collision_pairs, robot_model: str, ignore=("floor",)) -> list[str]:
    """Names of things the robot touched, from ros_gz_interfaces/Contacts pairs.

    A contact counts as a collision unless
    the other side is the robot itself or the floor. `collision_pairs` is an
    iterable of (collision1_name, collision2_name), e.g.
    [(c.collision1.name, c.collision2.name) for c in msg.contacts].
    Gazebo names look like "robot::base::body" / "north_wall::link::collision".
    """
    prefixes = (f"{robot_model}::",) + tuple(f"{name}::" for name in ignore)
    touched = []
    for pair in collision_pairs:
        for name in pair:
            if not name.startswith(prefixes) and name not in touched:
                touched.append(name)
    return touched


class SensorFreshness:
    """Tracks when each stream last arrived (wall clock).

    Use: call mark(name) in each subscription callback; after publishing an
    action, set barrier() and wait until ready() so the next state is built
    only from data that arrived after the action finished.
    """

    def __init__(self, names=("odom", "scan"), max_age_s: float = MAX_SENSOR_AGE_S,
                 clock=time.monotonic):
        self.names = tuple(names)
        self.max_age_s = max_age_s
        self.clock = clock
        self.last = {name: None for name in self.names}
        self.after = 0.0

    def mark(self, name: str) -> None:
        self.last[name] = self.clock()

    def barrier(self) -> None:
        self.after = self.clock()

    def ready(self) -> bool:
        now = self.clock()
        return all(
            t is not None and self.after <= t <= now and now - t <= self.max_age_s
            for t in self.last.values()
        )

    def ages(self) -> dict:
        now = self.clock()
        return {k: (None if t is None else round(now - t, 3)) for k, t in self.last.items()}


class ChaseEnv:
    """Planned wiring (to fill in once the NPC and reset exist in the sim):

    __init__: publisher /robot/cmd_vel; subscriptions /robot/odom, /robot/scan,
              /clock, /robot/contact, /npc/odom, each calling SensorFreshness.mark().
    step(a):  if the agent's action is FORWARD and behaviors.motion_unsafe() -> optional shield;
              publish the action's Twist until action_duration_s of sim time passes
              (same loop as sim/smoke_test.py drive()); publish zero Twist; barrier();
              spin until ready(); build rays with geometry.scan_to_rays(), sectors with
              geometry.clearances(); NPC via ros.detector.detect(); contact via
              external_contacts(); if a new collision, run behaviors.CollisionRecovery.plan().
    reset():  /world/room/set_pose for learner and NPC (needs the UserCommands
              system in room.sdf), restart the NPC route, clear trackers.
    """

    def __init__(self, node, cfg: dict):
        raise NotImplementedError

    def reset(self, spawn=None, route=None):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError
