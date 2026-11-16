"""Reward computation (ROS-free).

Stateful per episode so one-time bonuses (new region, first detection) can't be farmed.
Values come from config/rewards.yaml.
"""

# used when the config does not set them
DEFAULT_MIN_LOST_STEPS = 5
DEFAULT_MAX_REDISCOVERY_REWARDS = 3


# adds up the reward of every step and remembers what was already paid
class RewardTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.visited_regions = set()
        self.has_detected = False
        self.prev_visible = False
        self.prev_distance = None
        self.lost_steps = 0
        self.rediscovery_rewards = 0

    # accounts for the reset observation before any action is selected
    def begin(self, visible, distance, region_id):
        self.visited_regions.add(region_id)
        self.has_detected = visible
        self.prev_visible = visible
        self.prev_distance = None
        if visible:
            self.prev_distance = distance
        reward = 0.0
        if self.cfg.get("new_region_enabled", True):
            reward = self.cfg["new_region"]
        if visible:
            reward = reward + self.cfg["first_detection"]
        return reward

    # checks whether finding the NPC again earns the rediscovery bonus
    # brief visibility flicker does not count, and the bonus is capped per episode
    def rediscovery_earned(self):
        min_lost_steps = self.cfg.get(
            "rediscovery_min_lost_steps", DEFAULT_MIN_LOST_STEPS
        )
        if self.prev_visible:
            return False
        if self.lost_steps < min_lost_steps:
            return False
        if not self.cfg.get("rediscovery_enabled", True):
            return False
        max_rewards = self.cfg.get(
            "max_rediscovery_rewards", DEFAULT_MAX_REDISCOVERY_REWARDS
        )
        return self.rediscovery_rewards < max_rewards

    # returns the step reward and its parts, the parts are logged for the ablations
    def step(self, visible, distance, region_id, collided, captured):
        parts = {}
        parts["time_step"] = self.cfg["time_step"]

        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            if self.cfg.get("new_region_enabled", True):
                parts["new_region"] = self.cfg["new_region"]

        if visible:
            if not self.has_detected:
                parts["first_detection"] = self.cfg["first_detection"]
                self.has_detected = True
            elif self.rediscovery_earned():
                parts["rediscovery"] = self.cfg["rediscovery"]
                self.rediscovery_rewards += 1
            if self.prev_visible and self.prev_distance is not None:
                limit = self.cfg["approach_clip"]
                closed = self.prev_distance - distance
                parts["approach"] = max(-limit, min(limit, closed))
            self.lost_steps = 0
        elif self.has_detected:
            # lost penalties start only after the NPC has been seen once
            self.lost_steps += 1
            parts["lost_per_step"] = self.cfg["lost_per_step"]
            if self.prev_visible:
                parts["lose_target"] = self.cfg["lose_target"]

        if collided:
            parts["collision"] = self.cfg["collision"]
        if captured:
            parts["capture"] = self.cfg["capture"]

        self.prev_visible = visible
        self.prev_distance = None
        if visible:
            self.prev_distance = distance

        reward = float(sum(parts.values()))
        return reward, parts
