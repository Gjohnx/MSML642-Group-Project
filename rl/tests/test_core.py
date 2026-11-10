"""Unit tests for rl/core — no ROS or Gazebo needed: `python3 -m pytest rl/tests`."""

import pytest


@pytest.mark.skip(reason="TODO: implement state.encode")
def test_encode_is_unique_and_in_range():
    ...


@pytest.mark.skip(reason="TODO: implement RewardTracker")
def test_new_region_bonus_only_once_per_episode():
    ...


@pytest.mark.skip(reason="TODO: implement RewardTracker")
def test_approach_reward_needs_two_visible_steps():
    ...
