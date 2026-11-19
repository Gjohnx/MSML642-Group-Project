"""Per-episode metrics (ROS-free).

One accumulator per episode, fed by pose samples with sim-time stamps and
by per-step events: capture, time to first detection, time to capture,
collisions, rediscoveries, path length and return.
"""

import math


# collects the measurements of one episode
class EpisodeMetrics:
    def __init__(self, episode=0, seed=0, spawn="", route="", policy=""):
        # run labels
        self.episode = episode
        self.seed = seed
        self.spawn = spawn
        self.route = route
        self.policy = policy

        # step counters
        self.steps = 0
        self.episode_return = 0.0
        self.path_length_m = 0.0
        self.collisions = 0
        self.shielded_steps = 0
        self.search_steps = 0
        self.recovery_events = 0
        self.captured = False
        self.capture_step = None
        self.first_detection_step = None
        self.losses = 0
        self.rediscoveries = 0
        self.regions_visited = 0
        # capture / collisions / step_limit
        self.termination = ""

        # simulation times
        self.first_sim_s = None
        self.last_sim_s = None
        self.first_detection_sim_s = None
        self.capture_sim_s = None

        # bookkeeping, not part of the saved row
        self.previous_xy = None
        self.was_visible = False
        self.regions = set()

    # registers the reset observation without inventing an action at step zero
    def begin(self, visible, region, sim_s, reward):
        self.regions.add(region)
        self.regions_visited = len(self.regions)
        self.episode_return += reward
        self.was_visible = visible
        if visible:
            self.first_detection_step = 0
            self.first_detection_sim_s = sim_s

    # adds the distance moved since the last pose and tracks the sim time
    def observe_pose(self, x, y, sim_s=None):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("pose must be finite")
        if self.previous_xy is not None:
            dx = x - self.previous_xy[0]
            dy = y - self.previous_xy[1]
            self.path_length_m += math.hypot(dx, dy)
        self.previous_xy = (x, y)
        if sim_s is not None:
            if self.first_sim_s is None:
                self.first_sim_s = sim_s
            self.last_sim_s = sim_s

    # records the events of one step
    def observe_step(
        self, visible, reward, new_collision, captured, region=None, sim_s=None,
        shielded=False, search_override=False, recovered=False
    ):
        self.steps += 1
        self.shielded_steps += int(shielded)
        self.search_steps += int(search_override)
        self.recovery_events += int(recovered)
        self.episode_return += reward
        if new_collision:
            self.collisions += 1
        if region is not None and region not in self.regions:
            self.regions.add(region)
            self.regions_visited = len(self.regions)
        if visible and not self.was_visible:
            if self.first_detection_step is None:
                self.first_detection_step = self.steps
                self.first_detection_sim_s = sim_s
            else:
                self.rediscoveries += 1
        if self.was_visible and not visible:
            self.losses += 1
        self.was_visible = visible
        if captured and not self.captured:
            self.captured = True
            self.capture_step = self.steps
            self.capture_sim_s = sim_s

    # simulation time between the first and the last pose
    def sim_elapsed(self):
        if self.first_sim_s is None or self.last_sim_s is None:
            return None
        return max(0.0, self.last_sim_s - self.first_sim_s)

    # returns the row saved to episodes.csv and report.json
    def to_dict(self):
        data = {}
        data["episode"] = self.episode
        data["seed"] = self.seed
        data["spawn"] = self.spawn
        data["route"] = self.route
        data["policy"] = self.policy
        data["steps"] = self.steps
        data["episode_return"] = self.episode_return
        data["path_length_m"] = self.path_length_m
        data["collisions"] = self.collisions
        data["shielded_steps"] = self.shielded_steps
        data["search_steps"] = self.search_steps
        data["recovery_events"] = self.recovery_events
        data["captured"] = self.captured
        data["capture_step"] = self.capture_step
        data["first_detection_step"] = self.first_detection_step
        data["losses"] = self.losses
        data["rediscoveries"] = self.rediscoveries
        data["regions_visited"] = self.regions_visited
        data["termination"] = self.termination
        data["first_sim_s"] = self.first_sim_s
        data["last_sim_s"] = self.last_sim_s
        data["first_detection_sim_s"] = self.first_detection_sim_s
        data["capture_sim_s"] = self.capture_sim_s
        data["sim_elapsed_s"] = self.sim_elapsed()
        return data
