"""Per-episode metrics (ROS-free).

One accumulator per episode, fed by pose samples with sim-time stamps and
by per-step events: capture, time to first detection, time to capture,
collisions, rediscoveries, path length and return.
"""

import math
from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeMetrics:
    episode: int = 0
    seed: int = 0
    spawn: str = ""
    route: str = ""
    policy: str = ""

    steps: int = 0
    episode_return: float = 0.0
    path_length_m: float = 0.0
    collisions: int = 0
    captured: bool = False
    capture_step: int | None = None
    first_detection_step: int | None = None
    losses: int = 0
    rediscoveries: int = 0
    regions_visited: int = 0
    termination: str = ""             # capture / collisions / step_limit

    first_sim_s: float | None = None
    last_sim_s: float | None = None
    first_detection_sim_s: float | None = None
    capture_sim_s: float | None = None

    _previous_xy: tuple | None = field(default=None, repr=False)
    _was_visible: bool = field(default=False, repr=False)
    _regions: set = field(default_factory=set, repr=False)

    def begin(self, *, visible, region, sim_s, reward):
        """Register initial visibility without inventing an action at step zero."""
        self._regions.add(region)
        self.regions_visited = len(self._regions)
        self.episode_return += reward
        self._was_visible = visible
        if visible:
            self.first_detection_step = 0
            self.first_detection_sim_s = sim_s

    def observe_pose(self, x: float, y: float, sim_s: float | None = None) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("pose must be finite")
        if self._previous_xy is not None:
            self.path_length_m += math.hypot(x - self._previous_xy[0], y - self._previous_xy[1])
        self._previous_xy = (x, y)
        if sim_s is not None:
            if self.first_sim_s is None:
                self.first_sim_s = sim_s
            self.last_sim_s = sim_s

    def observe_step(self, *, visible: bool, reward: float, new_collision: bool,
                     captured: bool, region=None, sim_s: float | None = None) -> None:
        self.steps += 1
        self.episode_return += reward
        if new_collision:
            self.collisions += 1
        if region is not None and region not in self._regions:
            self._regions.add(region)
            self.regions_visited = len(self._regions)
        if visible and not self._was_visible:
            if self.first_detection_step is None:
                self.first_detection_step = self.steps
                self.first_detection_sim_s = sim_s
            else:
                self.rediscoveries += 1
        if self._was_visible and not visible:
            self.losses += 1
        self._was_visible = visible
        if captured and not self.captured:
            self.captured = True
            self.capture_step = self.steps
            self.capture_sim_s = sim_s

    @property
    def sim_elapsed_s(self) -> float | None:
        if self.first_sim_s is None or self.last_sim_s is None:
            return None
        return max(0.0, self.last_sim_s - self.first_sim_s)

    def to_dict(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        data["sim_elapsed_s"] = self.sim_elapsed_s
        return data
