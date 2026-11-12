"""Baseline policies (ROS-free). Same input as the agent: the discrete state tuple.

RandomPolicy      uniform over the 4 actions.
RuleBasedTracker  turn toward a visible NPC, avoid obstacles by turning toward
                  the side with more room, otherwise wander.
"""

import numpy as np

from rl.core.agent import ACTIONS, FORWARD, TURN_LEFT, TURN_RIGHT
from rl.core.state import LEFT, NEAR, RIGHT


class RandomPolicy:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, state: tuple) -> int:
        return int(self.rng.integers(len(ACTIONS)))


class RuleBasedTracker:
    def act(self, state: tuple) -> int:
        visible, direction, distance, front, left, right, _last_seen = state
        open_side = TURN_LEFT if left >= right else TURN_RIGHT
        if visible:
            if direction == LEFT:
                return TURN_LEFT
            if direction == RIGHT:
                return TURN_RIGHT
            # NPC dead ahead. If it is the near thing the lidar sees, drive at it.
            if front != NEAR or distance == NEAR:
                return FORWARD
            return open_side
        # Not visible: wander. (The env's LostTargetSearch override handles
        # turning toward the last-seen side for every policy alike.)
        return FORWARD if front != NEAR else open_side
