"""Tests for the explore-then-chase demo, using a small fake world instead of Gazebo.

Run from the repo root:  python3 -m pytest rl/tests
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl.core.agent import FORWARD, STOP, TURN_LEFT, TURN_RIGHT, QLearningAgent
from rl.core.baselines import RuleBasedTracker
from rl.core.explore import (
    CHASE,
    EXPLORE,
    ExploreRewardTracker,
    ExploreThenChase,
    OdometryAreas,
    encode_explore,
    explore_state,
    num_explore_states,
)
from rl.core.geometry import normalization, world_to_robot
from rl.core.state import LEFT, NONE, Observation, num_states
from rl.demo import ChaseButton, run_demo
from rl.explore import run_explore_episode, summarize_explore

CFG = yaml.safe_load(Path("rl/config/qlearning.yaml").read_text())
EXPLORE_CFG = yaml.safe_load(Path("rl/config/explore.yaml").read_text())

# fake world: an empty square room
ROOM_HALF_M = 5.0
WALL_MARGIN_M = 0.2
STEP_M = 0.3
TURN_RAD = 0.5
SEE_RANGE_M = 4.0
SEE_HALF_FOV = math.radians(45)
CAPTURE_M = 0.5
# the fake wheel odometry starts somewhere else than the true position
ODOMETRY_OFFSET = (12.0, -7.0)


# distance from a point to the room wall along a heading
def distance_to_wall(x, y, heading):
    best = math.inf
    dx = math.cos(heading)
    dy = math.sin(heading)
    if dx > 1e-9:
        best = min(best, (ROOM_HALF_M - x) / dx)
    if dx < -1e-9:
        best = min(best, (-ROOM_HALF_M - x) / dx)
    if dy > 1e-9:
        best = min(best, (ROOM_HALF_M - y) / dy)
    if dy < -1e-9:
        best = min(best, (-ROOM_HALF_M - y) / dy)
    return best


# stands in for ChaseEnv: same reset / observe / step / stop methods and info keys
class FakeEnv:
    def __init__(self, npc_xy, start=(0.0, 0.0, 0.0), max_steps=50, max_collisions=3):
        self.npc_xy = npc_xy
        self.start = start
        self.max_steps = max_steps
        self.max_collisions = max_collisions
        self.stopped = 0

    def reset(self, spawn=None, route=None):
        self.pose = self.start
        if spawn is not None:
            self.pose = tuple(spawn)
        self.steps = 0
        self.collisions = 0
        obs, info = self.observe()
        return obs

    # fake wheel odometry: the true position plus a fixed offset
    def odometry(self):
        return (self.pose[0] + ODOMETRY_OFFSET[0], self.pose[1] + ODOMETRY_OFFSET[1])

    # lidar ranges to the walls, and the NPC if it is in range and in view
    def observe(self):
        x, y, yaw = self.pose
        front = distance_to_wall(x, y, yaw)
        left = distance_to_wall(x, y, yaw + math.pi / 2)
        right = distance_to_wall(x, y, yaw - math.pi / 2)
        bearing, distance = world_to_robot(self.pose, self.npc_xy)
        visible = distance <= SEE_RANGE_M and abs(bearing) <= SEE_HALF_FOV
        if visible:
            obs = Observation(True, bearing, distance, front, left, right)
        else:
            obs = Observation(False, None, None, front, left, right)
        info = {}
        info["pose"] = self.pose
        info["npc_pose"] = (self.npc_xy[0], self.npc_xy[1], 0.0)
        info["sim_s"] = self.steps * 0.25
        info["region"] = (math.floor((x + 5) / 2), math.floor((y + 5) / 2))
        info["captured"] = distance <= CAPTURE_M
        info["collisions"] = self.collisions
        info["clearances"] = {
            "front": front,
            "left": left,
            "right": right,
            "rear": None,
        }
        return obs, info

    # moves the robot, a step into the wall counts as a collision and is not taken
    def step(self, action):
        x, y, yaw = self.pose
        new_collision = False
        if action == FORWARD:
            new_x = x + STEP_M * math.cos(yaw)
            new_y = y + STEP_M * math.sin(yaw)
            limit = ROOM_HALF_M - WALL_MARGIN_M
            if abs(new_x) > limit or abs(new_y) > limit:
                new_collision = True
                self.collisions += 1
            else:
                x = new_x
                y = new_y
        elif action == TURN_LEFT:
            yaw = normalization(yaw + TURN_RAD)
        elif action == TURN_RIGHT:
            yaw = normalization(yaw - TURN_RAD)
        self.pose = (x, y, yaw)
        self.steps += 1
        obs, info = self.observe()
        info["new_collision"] = new_collision
        info["steps"] = self.steps
        info["termination"] = ""
        if info["captured"]:
            info["termination"] = "capture"
        elif self.collisions >= self.max_collisions:
            info["termination"] = "collisions"
        elif self.steps >= self.max_steps:
            info["termination"] = "step_limit"
        info["done"] = bool(info["termination"])
        return obs, info

    # counts stop calls so the tests can check that episodes stop the robot
    def stop(self):
        self.stopped += 1


# a Q-agent whose table prefers one action in every state
def fixed_agent(n_states, action):
    agent = QLearningAgent(n_states, 0.1, 0.95, np.random.default_rng(0))
    agent.q[:, action] = 1.0
    return agent


# number of rows in the explore and chase Q-tables
def explore_count():
    return num_explore_states(CFG, EXPLORE_CFG)


def chase_count():
    return num_states(CFG)


# odometry memory


def test_odometry_areas_count_new_cells_and_stall():
    areas = OdometryAreas(1.0)
    assert areas.observe(0.2, 0.2)
    assert not areas.observe(0.8, 0.3)
    assert areas.steps_since_new == 1
    assert areas.observe(1.2, 0.3)
    assert areas.steps_since_new == 0
    assert len(areas.visited) == 2
    areas.reset()
    assert len(areas.visited) == 0


# explore state


def test_explore_encode_is_a_bijection():
    count = explore_count()
    assert count == 81
    seen = set()
    for front in range(3):
        for left in range(3):
            for right in range(3):
                for stall in range(3):
                    seen.add(
                        encode_explore((front, left, right, stall), CFG, EXPLORE_CFG)
                    )
    assert seen == set(range(count))


def test_explore_state_ignores_the_npc():
    near_npc = Observation(True, 0.0, 1.0, 0.3, 2.0, 2.0)
    no_npc = Observation(False, None, None, 0.3, 2.0, 2.0)
    assert explore_state(near_npc, 0, CFG, EXPLORE_CFG) == explore_state(
        no_npc, 0, CFG, EXPLORE_CFG
    )
    # stall levels: below 10 steps, 10 to 29, 30 and more
    assert explore_state(no_npc, 9, CFG, EXPLORE_CFG)[3] == 0
    assert explore_state(no_npc, 10, CFG, EXPLORE_CFG)[3] == 1
    assert explore_state(no_npc, 30, CFG, EXPLORE_CFG)[3] == 2


def test_explore_rewards():
    tracker = ExploreRewardTracker(EXPLORE_CFG["rewards"])
    tracker.begin((0, 0))
    reward, parts = tracker.step((0, 0), collided=False)
    assert parts == {"time_step": -0.02}
    reward, parts = tracker.step((1, 0), collided=False)
    assert reward == pytest.approx(0.18)
    reward, parts = tracker.step((1, 0), collided=True)
    assert reward == pytest.approx(-5.02)


# mode switch


def test_explore_table_is_used_until_chase_is_pressed():
    explore_agent = fixed_agent(explore_count(), TURN_RIGHT)
    chase_agent = fixed_agent(chase_count(), STOP)
    controller = ExploreThenChase(explore_agent, chase_agent, CFG, EXPLORE_CFG)
    npc_ahead = Observation(True, 0.0, 2.0, 3.0, 3.0, 3.0)
    assert controller.mode == EXPLORE
    # exploring ignores the NPC even when it is right ahead
    assert controller.choose_action(npc_ahead, {}) == TURN_RIGHT
    assert controller.request_chase(7)
    assert not controller.request_chase(9)
    assert controller.mode == CHASE and controller.switch_step == 7
    assert controller.choose_action(npc_ahead, {}) == STOP


def test_last_seen_from_exploring_guides_the_chase():
    explore_agent = fixed_agent(explore_count(), FORWARD)
    chase_agent = fixed_agent(chase_count(), FORWARD)
    controller = ExploreThenChase(explore_agent, chase_agent, CFG, EXPLORE_CFG)
    npc_on_left = Observation(True, math.radians(35), 2.0, 3.0, 3.0, 3.0)
    controller.choose_action(npc_on_left, {})
    assert controller.last_seen == LEFT
    controller.request_chase()
    hidden = Observation(False, None, None, 3.0, 3.0, 3.0)
    clear = {"front": 3.0, "left": 3.0, "right": 3.0}
    # out of view: the lost-target search turns toward the last-seen side
    assert controller.choose_action(hidden, clear) == TURN_LEFT


def test_reset_returns_to_exploring():
    controller = ExploreThenChase(
        fixed_agent(explore_count(), FORWARD), RuleBasedTracker(), CFG, EXPLORE_CFG
    )
    controller.choose_action(Observation(True, math.radians(35), 2.0, 3, 3, 3), {})
    controller.request_chase(3)
    controller.reset()
    assert controller.mode == EXPLORE
    assert controller.last_seen == NONE
    assert controller.switch_step is None


def test_chase_button():
    button = ChaseButton()
    assert not button.requested()
    button.receive(None)
    assert button.requested()
    automatic = ChaseButton(chase_after=2)
    results = []
    for i in range(4):
        results.append(automatic.requested())
    assert results == [False, False, True, True]


# full loops on the fake world


def test_explore_training_episode_runs_and_learns():
    env = FakeEnv(npc_xy=(4.0, 4.0), max_steps=40)
    agent = QLearningAgent(explore_count(), 0.1, 0.95, np.random.default_rng(1))
    row = run_explore_episode(
        env, agent, CFG, EXPLORE_CFG, env.odometry, None, None, epsilon=0.5, train=True
    )
    assert row["steps"] == 40 or row["termination"] == "collisions"
    assert agent.updates == row["steps"]
    assert row["regions_visited"] >= 1 and row["odometry_cells"] >= 1
    assert env.stopped == 1
    summary = summarize_explore([row, row])
    assert summary["episodes"] == 2


def test_explore_episode_does_not_end_on_touching_the_npc():
    # the NPC sits on the robot's path; while exploring that must not end the episode
    env = FakeEnv(npc_xy=(0.6, 0.0), max_steps=10)
    agent = fixed_agent(explore_count(), FORWARD)
    row = run_explore_episode(env, agent, CFG, EXPLORE_CFG, env.odometry, None, None)
    assert row["termination"] in ("step_limit", "collisions")
    assert row["npc_seen"]


def test_explore_learning_reduces_wall_crashes():
    env = FakeEnv(npc_xy=(9.0, 9.0), start=(3.5, 0.0, 0.0), max_steps=60)
    agent = QLearningAgent(explore_count(), 0.2, 0.95, np.random.default_rng(2))
    for episode in range(200):
        epsilon = max(0.05, 1.0 - episode / 150)
        run_explore_episode(
            env, agent, CFG, EXPLORE_CFG, env.odometry, None, None, epsilon, True
        )
    greedy = run_explore_episode(env, agent, CFG, EXPLORE_CFG, env.odometry, None, None)
    crash_into_wall = fixed_agent(explore_count(), FORWARD)
    naive = run_explore_episode(
        env, crash_into_wall, CFG, EXPLORE_CFG, env.odometry, None, None
    )
    assert naive["collisions"] == 3
    assert greedy["collisions"] < naive["collisions"]


def test_demo_explores_then_captures_after_chase_is_pressed():
    env = FakeEnv(npc_xy=(3.0, 0.0), max_steps=1000)
    # the explore table only spins in place, so it never reaches the NPC by itself
    controller = ExploreThenChase(
        fixed_agent(explore_count(), TURN_LEFT), RuleBasedTracker(), CFG, EXPLORE_CFG
    )
    button = ChaseButton(chase_after=8)
    records = []
    summary = run_demo(
        env, controller, env.odometry, button.requested, 200, records.append
    )
    assert summary["termination"] == "capture"
    assert summary["switch_step"] == 8
    assert summary["chase_steps"] > 0
    assert summary["npc_seen_while_exploring"]
    for record in records[:8]:
        assert record["mode"] == EXPLORE
    assert records[-1]["mode"] == CHASE


def test_demo_without_chase_keeps_exploring():
    # the NPC starts within capture distance, but without the button it is not a capture
    env = FakeEnv(npc_xy=(0.3, 0.0), max_steps=1000)
    controller = ExploreThenChase(
        fixed_agent(explore_count(), TURN_LEFT), RuleBasedTracker(), CFG, EXPLORE_CFG
    )
    summary = run_demo(env, controller, env.odometry, ChaseButton().requested, 12)
    assert summary["termination"] == "step_limit"
    assert summary["steps"] == 12
    assert summary["switch_step"] is None
