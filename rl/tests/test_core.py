"""Unit tests for rl/ — no ROS or Gazebo needed.

Run from the repo root:  python3 -m pytest rl/tests
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from rl.core import geometry
from rl.core.agent import (
    FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    QLearningAgent,
    epsilon_schedule,
)
from rl.core.baselines import RandomPolicy, RuleBasedTracker
from rl.core.behaviors import (
    CollisionCounter,
    CollisionRecovery,
    LostTargetSearch,
    SafetyConfig,
    motion_unsafe,
)
from rl.core.state import (
    FAR,
    FRONT,
    LEFT,
    MEDIUM,
    NEAR,
    NONE,
    RIGHT,
    Observation,
    discretize,
    encode,
    next_last_seen,
    num_states,
)
from rl.evaluate import summarize
from rl.metrics import EpisodeMetrics
from rl.ros.detector import detect
from rl.ros.env import SensorFreshness, external_contacts

CFG = yaml.safe_load(Path("rl/config/qlearning.yaml").read_text())

# fake lidar settings, same as /robot/scan
SCAN_RANGE_MIN = 0.05
SCAN_RANGE_MAX = 6.0

# detector settings used in the tests
DETECT_RANGE_M = 4.0
DETECT_FOV_DEG = 90


# builds a fake LaserScan with beam_count beams spread over fov_deg
# ranges_by_deg maps (low_deg, high_deg) to the range of the beams in that window
def make_scan(ranges_by_deg, beam_count=181, fov_deg=180.0):
    angle_min = -math.radians(fov_deg) / 2
    angle_increment = math.radians(fov_deg) / (beam_count - 1)
    ranges = []
    for i in range(beam_count):
        angle_deg = math.degrees(angle_min + i * angle_increment)
        value = math.inf
        for window in ranges_by_deg:
            low_deg, high_deg = window
            if low_deg <= angle_deg <= high_deg:
                value = ranges_by_deg[window]
        ranges.append(value)
    scan = SimpleNamespace()
    scan.angle_min = angle_min
    scan.angle_increment = angle_increment
    scan.range_min = SCAN_RANGE_MIN
    scan.range_max = SCAN_RANGE_MAX
    scan.ranges = ranges
    return scan


# distance covered by a backup command
def backup_distance(command):
    return -command.linear * command.duration_s


# geometry


def test_scan_to_rays_marks_misses_as_range_max():
    rays = geometry.scan_to_rays(make_scan({(-5, 5): 1.0}))
    assert len(rays) == 181
    for angle, distance, hit in rays:
        if hit:
            assert distance == 1.0
        else:
            assert distance == SCAN_RANGE_MAX


def test_clearances_sectors_and_missing_rear():
    scan = make_scan({(-5, 5): 0.8, (80, 90): 0.5, (-90, -80): 2.0})
    rays = geometry.scan_to_rays(scan)
    clear = geometry.clearances(rays, scan.range_max)
    assert clear["front"] == pytest.approx(0.8)
    assert clear["left"] == pytest.approx(0.5)
    assert clear["right"] == pytest.approx(2.0)
    # a 180 deg scan has no beams behind
    assert clear["rear"] is None


def test_world_robot_round_trip():
    pose = (1.0, 2.0, math.radians(30))
    world = geometry.robot_to_world(pose, (1.0, 0.5))
    bearing, distance = geometry.world_to_robot(pose, world)
    assert distance == pytest.approx(math.hypot(1.0, 0.5))
    assert bearing == pytest.approx(math.atan2(0.5, 1.0))


# state


# every state tuple must land on its own Q-table row
def test_num_states_and_encode_is_a_bijection():
    count = num_states(CFG)
    assert count == 1080
    seen = set()
    for visible in (0, 1):
        directions = (NONE,)
        distances = (NONE,)
        if visible:
            directions = (LEFT, FRONT, RIGHT)
            distances = (NEAR, MEDIUM, FAR)
        for direction in directions:
            for distance in distances:
                for last_seen in (LEFT, FRONT, RIGHT, NONE):
                    for front in range(3):
                        for left in range(3):
                            for right in range(3):
                                state = (
                                    visible,
                                    direction,
                                    distance,
                                    last_seen,
                                    front,
                                    left,
                                    right,
                                )
                                seen.add(encode(state, CFG))
    assert seen == set(range(count))


def test_discretize_bins_and_direction():
    obs = Observation(True, math.radians(30), 1.5, 0.3, 0.7, 3.0)
    state = discretize(obs, NONE, CFG)
    assert state == (1, LEFT, MEDIUM, NONE, NEAR, MEDIUM, FAR)
    hidden_obs = Observation(False, None, None, 2.0, 2.0, 2.0)
    hidden = discretize(hidden_obs, RIGHT, CFG)
    assert hidden == (0, NONE, NONE, RIGHT, FAR, FAR, FAR)
    assert next_last_seen(state, RIGHT) == LEFT
    assert next_last_seen(hidden, RIGHT) == RIGHT


# agent


def test_q_update_matches_formula():
    rng = np.random.default_rng(0)
    agent = QLearningAgent(5, alpha=0.1, gamma=0.95, rng=rng)
    agent.q[3] = [0.0, 2.0, 1.0, 0.0]
    agent.update(0, FORWARD, 1.0, 3, done=False)
    expected = 0.1 * (1.0 + 0.95 * 2.0)
    assert agent.q[0, FORWARD] == pytest.approx(expected)
    # a terminal step does not bootstrap from the next state
    agent.update(1, STOP, 30.0, 3, done=True)
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


# behaviors


def test_recovery_backs_up_when_rear_is_clear():
    plan = CollisionRecovery().plan({"rear": 1.0, "left": 0.4, "right": 2.0})
    # first command is a stop
    assert plan[0].linear == 0 and plan[0].angular == 0
    assert plan[1].linear < 0
    assert backup_distance(plan[1]) == pytest.approx(0.20)
    # right is more open
    assert plan[2].angular < 0


def test_recovery_uses_rear_offset_and_360_scan():
    scan = make_scan(
        {(170, 180): 0.9, (-180, -170): 0.9}, beam_count=360, fov_deg=359.0
    )
    rays = geometry.scan_to_rays(scan)
    clear = geometry.clearances(rays, scan.range_max)
    assert clear["rear"] == pytest.approx(0.9)
    recovery = CollisionRecovery(rear_offset_m=0.28)
    # 0.62 m free behind the body
    plan = recovery.plan({"rear": 0.9, "left": 1, "right": 1})
    assert backup_distance(plan[1]) == pytest.approx(0.20)
    # only 0.17 m free, no backup
    plan = recovery.plan({"rear": 0.45, "left": 1, "right": 1})
    for command in plan:
        assert command.linear == 0


def test_recovery_turns_in_place_without_rear_lidar():
    plan = CollisionRecovery().plan({"rear": None, "left": 1.0, "right": 0.5})
    for command in plan:
        assert command.linear == 0
    turn = plan[-1]
    assert turn.angular > 0
    assert turn.angular * turn.duration_s == pytest.approx(math.radians(35))


def test_recovery_short_backup_rule():
    # backs up rear - 0.12
    plan = CollisionRecovery().plan({"rear": 0.27, "left": 1, "right": 1})
    assert backup_distance(plan[1]) == pytest.approx(0.15)
    # too tight, no backup
    plan = CollisionRecovery().plan({"rear": 0.20, "left": 1, "right": 1})
    assert len(plan) == 2


def test_collision_counter_counts_separate_contacts():
    counter = CollisionCounter(3)
    flags = [True, True, False, True, False, False, True]
    new_collisions = []
    for flag in flags:
        new_collisions.append(counter.observe(flag))
    assert new_collisions == [True, False, False, True, False, False, True]
    assert counter.count == 3 and counter.done


def test_motion_unsafe():
    safety = SafetyConfig()
    near_wall = geometry.scan_to_rays(make_scan({(-10, 10): 0.15}))
    assert motion_unsafe(near_wall, SCAN_RANGE_MAX, 0.0, safety)
    open_space = geometry.scan_to_rays(make_scan({}))
    assert not motion_unsafe(open_space, SCAN_RANGE_MAX, 0.0, safety)


def test_lost_target_search_scans_then_explores():
    search = LostTargetSearch(scan_steps=2)
    clear = {"front": 0.5, "left": 2.0, "right": 0.5}
    assert search.next_action(RIGHT, clear) == TURN_RIGHT
    assert search.next_action(RIGHT, clear) == TURN_RIGHT
    # explore: front blocked, left open
    assert search.next_action(RIGHT, clear) == TURN_LEFT
    assert search.next_action(RIGHT, {"front": 3.0}) == FORWARD


# baselines


def test_rule_based_tracker():
    rule = RuleBasedTracker()
    assert rule.act((1, LEFT, FAR, NONE, FAR, FAR, FAR)) == TURN_LEFT
    # the near thing ahead is the NPC
    assert rule.act((1, FRONT, NEAR, FRONT, NEAR, FAR, FAR)) == FORWARD
    assert rule.act((0, NONE, NONE, NONE, NEAR, NEAR, FAR)) == TURN_RIGHT
    # lost: turn to the last-seen side
    assert rule.act((0, NONE, NONE, LEFT, FAR, FAR, FAR)) == TURN_LEFT
    assert rule.act((0, NONE, NONE, NONE, FAR, FAR, FAR)) == FORWARD

    random_policy = RandomPolicy(np.random.default_rng(0))
    actions = set()
    for i in range(200):
        actions.add(random_policy.act(None))
    assert actions == {0, 1, 2, 3}


# detector and env helpers


# returns only the visible flag of the detector
def npc_visible(pose, npc_xy, rays):
    visible, bearing, distance = detect(
        pose, npc_xy, rays, DETECT_RANGE_M, DETECT_FOV_DEG
    )
    return visible


def test_detector_range_fov_and_occlusion():
    pose = (0.0, 0.0, 0.0)
    clear = geometry.scan_to_rays(make_scan({}))
    assert npc_visible(pose, (2.0, 0.0), clear)
    # out of range
    assert not npc_visible(pose, (5.0, 0.0), clear)
    # outside the field of view
    assert not npc_visible(pose, (0.0, 2.0), clear)
    # behind a wall
    wall = geometry.scan_to_rays(make_scan({(-5, 5): 0.8}))
    assert not npc_visible(pose, (3.0, 0.0), wall)
    # the lidar hits the NPC itself
    npc_itself = geometry.scan_to_rays(make_scan({(-5, 5): 1.72}))
    assert npc_visible(pose, (2.0, 0.0), npc_itself)


def test_external_contacts_ignores_self_and_floor():
    floor_pair = ("robot::base::body", "floor::link::collision")
    wall_pair = ("robot::base::body", "north_wall::link::collision")
    touched = external_contacts([floor_pair, wall_pair], "robot")
    assert touched == ["north_wall::link::collision"]
    assert external_contacts([floor_pair], "robot") == []


def test_sensor_freshness():
    now = [0.0]

    def fake_clock():
        return now[0]

    fresh = SensorFreshness(("odom", "scan"), max_age_s=2.0, clock=fake_clock)
    fresh.mark("odom")
    assert not fresh.ready()
    now[0] = 1.0
    fresh.mark("scan")
    assert fresh.ready()
    fresh.barrier()
    now[0] = 1.5
    # data is older than the barrier
    assert not fresh.ready()
    fresh.mark("odom")
    fresh.mark("scan")
    assert fresh.ready()


# metrics


def test_episode_metrics_and_summary():
    metrics = EpisodeMetrics()
    metrics.observe_pose(0, 0, sim_s=10.0)
    metrics.observe_pose(3, 4, sim_s=12.0)
    # (visible, captured) for each step
    steps = [
        (False, False),
        (True, False),
        (False, False),
        (True, False),
        (True, True),
    ]
    for visible, captured in steps:
        metrics.observe_step(
            visible=visible,
            reward=1.0,
            new_collision=False,
            captured=captured,
            region=(0, 0),
        )
    row = metrics.to_dict()
    assert row["path_length_m"] == 5.0
    assert row["sim_elapsed_s"] == 2.0
    assert row["first_detection_step"] == 2
    assert row["losses"] == 1
    assert row["rediscoveries"] == 1
    assert row["captured"]
    assert row["capture_step"] == 5
    assert row["regions_visited"] == 1

    empty_row = EpisodeMetrics().to_dict()
    summary = summarize([row, empty_row])
    assert summary["capture_rate"] == 0.5
    assert summary["rediscovery_rate"] == 1.0
