"""Explore-then-chase demo logic (ROS-free).

The robot first explores with its own small Q-table, which only sees the lidar
obstacle levels and how long it has been since it reached a new area. When a
person clicks the "Chase" button, it switches to the trained chase Q-table.
The direction where the NPC was last seen is tracked during exploration too,
so the chase starts knowing which way to turn.

Explore state tuple (all ints):
    obstacle_front : 0 = near .. k-1 = far   (same bins as the chase state)
    obstacle_left  : same
    obstacle_right : same
    stall          : 0 = moving on, 1 = slowing down, 2 = stuck
"""

import math

from rl.core.agent import QLearningAgent
from rl.core.behaviors import LostTargetSearch
from rl.core.state import NONE, bin_value, discretize, encode, next_last_seen

# robot modes in the demo
EXPLORE = "explore"
CHASE = "chase"

# used when the explore config does not set them
DEFAULT_ODOMETRY_CELL_M = 1.0
DEFAULT_STALL_BINS = [10, 30]


# remembers which odometry cells the robot has driven through this episode
# the cells come from the robot's own wheel odometry, not the true position
class OdometryAreas:
    def __init__(self, cell_m=DEFAULT_ODOMETRY_CELL_M):
        self.cell_m = cell_m
        self.reset()

    def reset(self):
        self.visited = set()
        self.steps_since_new = 0

    # returns the odometry cell that contains a point
    def cell(self, x, y):
        cell_x = math.floor(x / self.cell_m)
        cell_y = math.floor(y / self.cell_m)
        return (cell_x, cell_y)

    # call once per step, returns True when the robot entered a new cell
    def observe(self, x, y):
        cell = self.cell(x, y)
        if cell in self.visited:
            self.steps_since_new += 1
            return False
        self.visited.add(cell)
        self.steps_since_new = 0
        return True


# returns how many obstacle levels and stall levels the configs define
def explore_level_counts(cfg, explore_cfg):
    obstacle_levels = len(cfg["obstacle_bins_m"]) + 1
    stall_levels = len(explore_cfg.get("stall_bins_steps", DEFAULT_STALL_BINS)) + 1
    return obstacle_levels, stall_levels


# turns an observation and the stall counter into the explore state tuple
def explore_state(obs, steps_since_new, cfg, explore_cfg):
    obstacle_edges = cfg["obstacle_bins_m"]
    stall_edges = explore_cfg.get("stall_bins_steps", DEFAULT_STALL_BINS)
    front = bin_value(obs.range_front_m, obstacle_edges)
    left = bin_value(obs.range_left_m, obstacle_edges)
    right = bin_value(obs.range_right_m, obstacle_edges)
    stall = bin_value(steps_since_new, stall_edges)
    return (front, left, right, stall)


# number of rows in the explore Q-table
def num_explore_states(cfg, explore_cfg):
    obstacle_levels, stall_levels = explore_level_counts(cfg, explore_cfg)
    return obstacle_levels**3 * stall_levels


# maps an explore state tuple to its row in the explore Q-table
def encode_explore(state, cfg, explore_cfg):
    obstacle_levels, stall_levels = explore_level_counts(cfg, explore_cfg)
    front, left, right, stall = state
    index = front
    index = index * obstacle_levels + left
    index = index * obstacle_levels + right
    index = index * stall_levels + stall
    return index


# reward for exploring: new map cells, a small cost per step, and collisions
# the cells use the true position, which is allowed for rewards but never for the state
class ExploreRewardTracker:
    def __init__(self, rewards):
        self.rewards = rewards
        self.reset()

    def reset(self):
        self.visited_regions = set()

    # the start cell is already visited, so it earns nothing
    def begin(self, region_id):
        self.visited_regions.add(region_id)

    # returns the step reward and its parts
    def step(self, region_id, collided):
        parts = {}
        parts["time_step"] = self.rewards["time_step"]
        if region_id not in self.visited_regions:
            self.visited_regions.add(region_id)
            parts["new_region"] = self.rewards["new_region"]
        if collided:
            parts["collision"] = self.rewards["collision"]
        reward = float(sum(parts.values()))
        return reward, parts


# picks actions for the demo: the explore table first, the chase policy after the click
# chase_policy is a trained QLearningAgent or a baseline with act(state)
class ExploreThenChase:
    def __init__(self, explore_agent, chase_policy, cfg, explore_cfg):
        self.explore_agent = explore_agent
        self.chase_policy = chase_policy
        self.cfg = cfg
        self.explore_cfg = explore_cfg
        cell_m = explore_cfg.get("odometry_cell_m", DEFAULT_ODOMETRY_CELL_M)
        self.areas = OdometryAreas(cell_m)
        self.search = LostTargetSearch(**cfg.get("lost_search", {}))
        self.reset()

    def reset(self):
        self.mode = EXPLORE
        self.last_seen = NONE
        self.areas.reset()
        self.search.reset()
        self.switch_step = None

    # the Chase button was pressed, returns True only the first time
    def request_chase(self, step=None):
        if self.mode == CHASE:
            return False
        self.mode = CHASE
        self.switch_step = step
        self.search.reset()
        return True

    # updates the odometry memory, call once after every step
    def observe_odometry(self, x, y):
        return self.areas.observe(x, y)

    # chooses the next action and updates the last-seen direction
    # clear is geometry.clearances() of the latest scan, used by the lost-target search
    def choose_action(self, obs, clear):
        chase_state = discretize(obs, self.last_seen, self.cfg)
        if self.mode == EXPLORE:
            action = self.explore_action(obs)
        else:
            action = self.chase_action(obs, chase_state, clear)
        self.last_seen = next_last_seen(chase_state, self.last_seen)
        return action

    # best action of the explore table for the current obstacles and stall level
    def explore_action(self, obs):
        state = explore_state(
            obs, self.areas.steps_since_new, self.cfg, self.explore_cfg
        )
        row = encode_explore(state, self.cfg, self.explore_cfg)
        if not isinstance(self.explore_agent, QLearningAgent):
            return self.explore_agent.act((0, NONE, NONE, NONE, *state[:3]))
        return self.explore_agent.greedy(row)

    # same rule as the training runner: the shared lost-target search takes over
    # while the NPC is out of view and its last direction is known
    def chase_action(self, obs, chase_state, clear):
        search_enabled = self.cfg.get("lost_search_enabled", True)
        if not obs.npc_visible and self.last_seen != NONE and search_enabled:
            return self.search.next_action(self.last_seen, clear)
        self.search.reset()
        if isinstance(self.chase_policy, QLearningAgent):
            return self.chase_policy.greedy(encode(chase_state, self.cfg))
        return self.chase_policy.act(chase_state)
