"""ROS 2 <-> RL bridge for the sim in sim/ (the only module in rl/ that talks to ROS).

Topics (see sim/bridge.yaml):
    /robot/cmd_vel  Twist      publish the chosen action for action_duration_s of sim time
    /robot/odom     Odometry   learner pose / timing (same pattern as sim/smoke_test.py)
    /robot/scan     LaserScan  181 beams, -90..+90 deg, 0.05-6.0 m
    /npc/odom       Odometry   needs NPC model + bridge entry (not in sim yet)

Interface:
    reset(spawn=None, route=None) -> Observation
    step(action) -> (Observation, info)

`info` holds ground truth (learner pose, true NPC pose, region id, collided,
captured). It is used for rewards and metrics only and never goes into the state.

Open items:
    - Reset without restarting the container: call Gazebo's /world/room/set_pose
      service for the learner and NPC (today the README uses `docker compose restart sim`).
    - Collision: no contact sensor in room.sdf yet, so start with
      min(scan) < collision_range_m; swap for a contact sensor later if needed.
"""

from rl.core.state import Observation  # noqa: F401


class ChaseEnv:
    def __init__(self, node, cfg: dict):
        raise NotImplementedError

    def reset(self, spawn=None, route=None):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError
