"""Baseline policies for comparison (ROS-free). Same act(state) interface as the agent."""


class RandomPolicy:
    def act(self, state: tuple) -> int:
        raise NotImplementedError


class RuleBasedTracker:
    """Turn toward a visible NPC and drive forward; avoid obstacles; otherwise wander."""

    def act(self, state: tuple) -> int:
        raise NotImplementedError
