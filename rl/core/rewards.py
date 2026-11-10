"""Reward computation (ROS-free).

Stateful per episode so one-time bonuses (new region, first detection) can't be farmed.
Values come from config/rewards.yaml.
"""


class RewardTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.visited_regions: set = set()
        self.has_detected = False
        self.prev_visible = False
        self.prev_distance: float | None = None

    def step(self, *, visible: bool, distance: float | None, region_id,
             collided: bool, captured: bool) -> tuple[float, dict]:
        """Return (reward, breakdown) — breakdown is logged for reward ablations."""
        raise NotImplementedError
