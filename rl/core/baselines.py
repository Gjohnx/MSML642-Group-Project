"""Baseline policies (ROS-free). Same input as the agent: the discrete state tuple.

RandomPolicy      uniform over the 4 actions.
RuleBasedTracker  turn toward a visible NPC, avoid obstacles by turning toward
                  the side with more room, turn toward the last-seen side
                  when the NPC is lost, otherwise drive forward.
"""

from rl.core.agent import ACTIONS, FORWARD, TURN_LEFT, TURN_RIGHT
from rl.core.state import LEFT, NEAR, RIGHT


# picks one of the four actions at random
class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, state):
        action = self.rng.integers(len(ACTIONS))
        return int(action)


# hand-written tracker used as a baseline
class RuleBasedTracker:
    def act(self, state):
        visible, direction, distance, last_seen, front, left, right = state
        open_side = TURN_RIGHT
        if left >= right:
            open_side = TURN_LEFT
        if visible:
            if direction == LEFT:
                return TURN_LEFT
            if direction == RIGHT:
                return TURN_RIGHT
            # NPC dead ahead, if it is the near thing the lidar sees, drive at it
            if front != NEAR or distance == NEAR:
                return FORWARD
            return open_side
        if front == NEAR:
            return open_side
        # lost: turn toward where the NPC was last seen
        # this only decides the action when the shared lost-target search is off
        if last_seen == LEFT:
            return TURN_LEFT
        if last_seen == RIGHT:
            return TURN_RIGHT
        return FORWARD
