"""Unit tests for rl/ — no ROS or Gazebo needed.

Run from the repo root:  python3 -m pytest rl/tests
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rl.core import geometry as g
from rl.core.agent import FORWARD, STOP, TURN_LEFT, TURN_RIGHT, QLearningAgent, epsilon_schedule
from rl.core.baselines import RandomPolicy, RuleBasedTracker
from rl.core.behaviors import (CollisionCounter, CollisionRecovery, LostTargetSearch,
                               SafetyConfig, motion_unsafe)
from rl.core.state import (FAR, FRONT, LEFT, MEDIUM, NEAR, NONE, RIGHT, Observation,
                           discretize, encode, next_last_seen, num_states)
from rl.evaluate import summarize
from rl.metrics import EpisodeMetrics
from rl.ros.detector import detect
from rl.ros.env import SensorFreshness, external_contacts

CFG = yaml.safe_load(open("rl/config/qlearning.yaml"))


def make_scan(ranges_by_deg, n=181, fov_deg=180.0, range_max=6.0, default=math.inf):
    """Fake LaserScan like /robot/scan: n beams over +-fov/2. ranges_by_deg maps
    (lo_deg, hi_deg) -> range for beams in that window."""
    angle_min = -math.radians(fov_deg) / 2
    inc = math.radians(fov_deg) / (n - 1)
    ranges = []
    for i in range(n):
        deg = math.degrees(angle_min + i * inc)
        r = default
        for (lo, hi), value in ranges_by_deg.items():
            if lo <= deg <= hi:
                r = value
        ranges.append(r)
    return SimpleNamespace(angle_min=angle_min, angle_increment=inc, range_min=0.05,
                           range_max=range_max, ranges=ranges)


# ---------------------------------------------------------------- geometry

def test_scan_to_rays_marks_misses_as_range_max():
    rays = g.scan_to_rays(make_scan({(-5, 5): 1.0}))
    assert len(rays) == 181
    hits = [r for r in rays if r[2]]
    assert all(r[1] == 1.0 for r in hits)
    assert all(r[1] == 6.0 for r in rays if not r[2])


def test_clearances_sectors_and_missing_rear():
    scan = make_scan({(-5, 5): 0.8, (80, 90): 0.5, (-90, -80): 2.0})
    c = g.clearances(g.scan_to_rays(scan), scan.range_max)
    assert c["front"] == pytest.approx(0.8)
    assert c["left"] == pytest.approx(0.5)
    assert c["right"] == pytest.approx(2.0)
    assert c["rear"] is None          # 180 deg lidar: no beams behind


def test_world_robot_round_trip():
    pose = (1.0, 2.0, math.radians(30))
    world = g.robot_to_world(pose, (1.0, 0.5))
    bearing, distance = g.world_to_robot(pose, world)
    assert distance == pytest.approx(math.hypot(1.0, 0.5))
    assert bearing == pytest.approx(math.atan2(0.5, 1.0))


# ---------------------------------------------------------------- state

def test_num_states_and_encode_is_a_bijection():
    n = num_states(CFG)
    assert n == 1080
    seen = set()
    for visible in (0, 1):
        dirs = (LEFT, FRONT, RIGHT) if visible else (NONE,)
        dists = (NEAR, MEDIUM, FAR) if visible else (NONE,)
        for d in dirs:
            for dist in dists:
                for f in range(3):
                    for l in range(3):
                        for r in range(3):
                            for last in (LEFT, FRONT, RIGHT, NONE):
                                seen.add(encode((visible, d, dist, f, l, r, last), CFG))
    assert seen == set(range(n))


def test_discretize_bins_and_direction():
    obs = Observation(True, math.radians(30), 1.5, 0.3, 0.7, 3.0)
    s = discretize(obs, NONE, CFG)
    assert s == (1, LEFT, MEDIUM, NEAR, MEDIUM, FAR, NONE)
    hidden = discretize(Observation(False, None, None, 2.0, 2.0, 2.0), RIGHT, CFG)
    assert hidden == (0, NONE, NONE, FAR, FAR, FAR, RIGHT)
    assert next_last_seen(s, RIGHT) == LEFT
    assert next_last_seen(hidden, RIGHT) == RIGHT


# ---------------------------------------------------------------- agent

def test_q_update_matches_formula():
    agent = QLearningAgent(5, alpha=0.1, gamma=0.95, rng=np.random.default_rng(0))
    agent.q[3] = [0.0, 2.0, 1.0, 0.0]
    agent.update(0, FORWARD, 1.0, 3, done=False)
    assert agent.q[0, FORWARD] == pytest.approx(0.1 * (1.0 + 0.95 * 2.0))
    agent.update(1, STOP, 30.0, 3, done=True)          # terminal: no bootstrap
    assert agent.q[1, STOP] == pytest.approx(3.0)


def test_greedy_and_epsilon_schedule(tmp_path):
    agent = QLearningAgent(2, 0.1, 0.95, np.random.default_rng(0))
    agent.q[0] = [0, 0, 5, 0]
    assert agent.act(0, epsilon=0.0) == (TURN_RIGHT, False)
    assert epsilon_schedule(0, 1.0, 0.05, 1500) == 1.0
    assert epsilon_schedule(750, 1.0, 0.05, 1500) == pytest.approx(0.525)
    assert epsilon_schedule(5000, 1.0, 0.05, 1500) == 0.05
    path = tmp_path / "q.npy"
    agent.save(path)
    other = QLearningAgent(2, 0.1, 0.95, np.random.default_rng(1))
    other.load(path)
    assert np.array_equal(other.q, agent.q)


# ---------------------------------------------------------------- behaviors

def test_recovery_backs_up_when_rear_is_clear():
    plan = CollisionRecovery().plan({"rear": 1.0, "left": 0.4, "right": 2.0})
    assert plan[0].linear == 0 and plan[0].angular == 0            # stop
    assert plan[1].linear < 0
    assert -plan[1].linear * plan[1].duration_s == pytest.approx(0.20)   # 0.2 m
    assert plan[2].angular < 0                                     # right is more open


def test_recovery_turns_in_place_without_rear_lidar():
    plan = CollisionRecovery().plan({"rear": None, "left": 1.0, "right": 0.5})
    assert all(c.linear == 0 for c in plan)
    assert plan[-1].angular > 0
    assert plan[-1].angular * plan[-1].duration_s == pytest.approx(math.radians(35))


def test_recovery_short_backup_rule():
    plan = CollisionRecovery().plan({"rear": 0.27, "left": 1, "right": 1})
    assert -plan[1].linear * plan[1].duration_s == pytest.approx(0.15)   # rear - 0.12
    plan = CollisionRecovery().plan({"rear": 0.20, "left": 1, "right": 1})
    assert len(plan) == 2                                          # too tight: no backup


def test_collision_counter_counts_separate_contacts():
    counter = CollisionCounter(3)
    flags = [True, True, False, True, False, False, True]
    new = [counter.observe(f) for f in flags]
    assert new == [True, False, False, True, False, False, True]
    assert counter.count == 3 and counter.done


def test_motion_unsafe():
    safety = SafetyConfig()
    near_wall = g.scan_to_rays(make_scan({(-10, 10): 0.15}))
    assert motion_unsafe(near_wall, 6.0, 0.0, safety)
    open_space = g.scan_to_rays(make_scan({}))
    assert not motion_unsafe(open_space, 6.0, 0.0, safety)


def test_lost_target_search_scans_then_explores():
    search = LostTargetSearch(scan_steps=2)
    clear = {"front": 0.5, "left": 2.0, "right": 0.5}
    assert search.next_action(RIGHT, clear) == TURN_RIGHT
    assert search.next_action(RIGHT, clear) == TURN_RIGHT
    assert search.next_action(RIGHT, clear) == TURN_LEFT   # explore: front blocked, left open
    assert search.next_action(RIGHT, {"front": 3.0}) == FORWARD


# ---------------------------------------------------------------- baselines

def test_rule_based_tracker():
    rule = RuleBasedTracker()
    assert rule.act((1, LEFT, FAR, FAR, FAR, FAR, NONE)) == TURN_LEFT
    assert rule.act((1, FRONT, NEAR, NEAR, FAR, FAR, FRONT)) == FORWARD   # the near thing is the NPC
    assert rule.act((0, NONE, NONE, NEAR, NEAR, FAR, NONE)) == TURN_RIGHT
    rng_policy = RandomPolicy(np.random.default_rng(0))
    assert {rng_policy.act(None) for _ in range(200)} == {0, 1, 2, 3}


# ---------------------------------------------------------------- detector / env helpers

def test_detector_range_fov_and_occlusion():
    pose = (0.0, 0.0, 0.0)
    clear = g.scan_to_rays(make_scan({}))
    assert detect(pose, (2.0, 0.0), clear, 4.0, 90)[0]
    assert not detect(pose, (5.0, 0.0), clear, 4.0, 90)[0]          # out of range
    assert not detect(pose, (0.0, 2.0), clear, 4.0, 90)[0]          # outside FOV
    wall = g.scan_to_rays(make_scan({(-5, 5): 0.8}))
    assert not detect(pose, (3.0, 0.0), wall, 4.0, 90)[0]           # behind a wall
    npc_itself = g.scan_to_rays(make_scan({(-5, 5): 1.72}))
    assert detect(pose, (2.0, 0.0), npc_itself, 4.0, 90)[0]         # lidar hits the NPC


def test_external_contacts_ignores_self_and_floor():
    pairs = [("robot::base::body", "floor::link::collision"),
             ("robot::base::body", "north_wall::link::collision")]
    assert external_contacts(pairs, "robot") == ["north_wall::link::collision"]
    assert external_contacts(pairs[:1], "robot") == []


def test_sensor_freshness():
    now = [0.0]
    fresh = SensorFreshness(("odom", "scan"), max_age_s=2.0, clock=lambda: now[0])
    fresh.mark("odom")
    assert not fresh.ready()
    now[0] = 1.0
    fresh.mark("scan")
    assert fresh.ready()
    fresh.barrier()
    now[0] = 1.5
    assert not fresh.ready()            # data is older than the barrier
    fresh.mark("odom"); fresh.mark("scan")
    assert fresh.ready()


# ---------------------------------------------------------------- metrics

def test_episode_metrics_and_summary():
    m = EpisodeMetrics()
    m.observe_pose(0, 0, sim_s=10.0)
    m.observe_pose(3, 4, sim_s=12.0)
    for vis, cap in [(False, False), (True, False), (False, False), (True, False), (True, True)]:
        m.observe_step(visible=vis, reward=1.0, new_collision=False, captured=cap, region=(0, 0))
    d = m.to_dict()
    assert d["path_length_m"] == 5.0 and d["sim_elapsed_s"] == 2.0
    assert (d["first_detection_step"], d["losses"], d["rediscoveries"]) == (2, 1, 1)
    assert d["captured"] and d["capture_step"] == 5 and d["regions_visited"] == 1
    s = summarize([d, EpisodeMetrics().to_dict()])
    assert s["capture_rate"] == 0.5 and s["rediscovery_rate"] == 1.0
