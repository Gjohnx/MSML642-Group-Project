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
        self.lost_steps = 0
        self.rediscovery_rewards = 0

    def begin(self, *, visible, distance, region_id):
        """Account for the reset observation before any action is selected."""
        self.visited_regions.add(region_id)
        self.has_detected = self.prev_visible = visible
        self.prev_distance = distance if visible else None
        reward = self.cfg["new_region"] if self.cfg.get("new_region_enabled", True) else 0.0
        return reward + (self.cfg["first_detection"] if visible else 0.0)

    def step(self, *, visible: bool, distance: float | None, region_id,
             collided: bool, captured: bool) -> tuple[float, dict]:
        """Return (reward, breakdown) — breakdown is logged for reward ablations."""
        parts = {"time_step": self.cfg["time_step"]}
        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            if self.cfg.get("new_region_enabled", True):
                parts["new_region"] = self.cfg["new_region"]
        if visible:
            if not self.has_detected:
                parts["first_detection"] = self.cfg["first_detection"]
                self.has_detected = True
            elif not self.prev_visible and self.lost_steps >= self.cfg.get("rediscovery_min_lost_steps", 5):
                if (self.cfg.get("rediscovery_enabled", True)
                        and self.rediscovery_rewards < self.cfg.get("max_rediscovery_rewards", 3)):
                    parts["rediscovery"] = self.cfg["rediscovery"]
                    self.rediscovery_rewards += 1
            if self.prev_visible and self.prev_distance is not None:
                limit = self.cfg["approach_clip"]
                parts["approach"] = max(-limit, min(limit, self.prev_distance - distance))
            self.lost_steps = 0
        else:
            self.lost_steps += 1
            parts["lost_per_step"] = self.cfg["lost_per_step"]
            if self.prev_visible:
                parts["lose_target"] = self.cfg["lose_target"]
        if collided:
            parts["collision"] = self.cfg["collision"]
        if captured:
            parts["capture"] = self.cfg["capture"]
        self.prev_visible = visible
        self.prev_distance = distance if visible else None
        return float(sum(parts.values())), parts
