"""Limited-range, limited-FOV NPC detector.

Uses ground-truth learner and NPC poses, but only reports the NPC when it is
within detector.range_m, inside detector.fov_deg, and not hidden behind an
obstacle (compare NPC distance with the /robot/scan beam at that bearing).
Returns (visible, bearing_rad, distance_m) in the learner's frame.
"""


def detect(learner_pose, npc_pose, scan, range_m: float, fov_deg: float):
    raise NotImplementedError
