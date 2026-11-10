"""Rule-based behaviors that override the Q-policy (ROS-free).

These return low-level action sequences; the env executes them before control
returns to the agent. They are NOT extra Q-learning actions.
"""


class CollisionRecovery:
    """stop -> back up ~0.2 m if rear is clear -> turn toward open side.
    If backing up is unsafe: stop and turn in place."""

    def plan(self, range_rear_m: float, range_left_m: float, range_right_m: float) -> list:
        raise NotImplementedError


class LostTargetSearch:
    """Turn toward last-seen direction and scan; if still not visible,
    explore using local obstacle readings until NPC is re-acquired."""

    def plan(self, last_seen_dir: int, ranges: dict) -> list:
        raise NotImplementedError
