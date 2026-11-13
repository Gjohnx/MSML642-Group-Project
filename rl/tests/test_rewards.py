"""Regression checks for proposal rewards and hidden-target information limits."""
import yaml
from pathlib import Path
from rl.core.rewards import RewardTracker


def tracker():
    return RewardTracker(yaml.safe_load(Path("rl/config/rewards.yaml").read_text()))


def step(t, visible=False, distance=None, region=(0, 0), collided=False, captured=False):
    return t.step(visible=visible, distance=distance, region_id=region,
                  collided=collided, captured=captured)[1]


def test_region_detection_and_approach():
    t = tracker()
    assert step(t)["new_region"] == 0.2
    assert "new_region" not in step(t)
    assert step(t, True, 3)["first_detection"] == 5
    assert step(t, True, 1)["approach"] == 1
    assert step(t)["lose_target"] == -1
    assert "approach" not in step(t, True, 0.8)


def test_visibility_flicker_and_rediscovery_budget():
    t = tracker()
    step(t, True, 2)
    for _ in range(10):
        step(t)
        assert "rediscovery" not in step(t, True, 2)
    earned = 0
    for _ in range(8):
        for _ in range(5):
            step(t)
        earned += int("rediscovery" in step(t, True, 2))
    assert earned == 3
    t.reset()
    assert step(t, True, 2)["first_detection"] == 5


def test_ablation_flags_and_terminal_rewards():
    t = tracker()
    t.cfg["new_region_enabled"] = False
    t.cfg["rediscovery_enabled"] = False
    assert "new_region" not in step(t)
    parts = step(t, collided=True, captured=True)
    assert parts["collision"] == -5
    assert parts["capture"] == 30


def test_reset_detection_is_not_credited_to_first_action():
    t = tracker()
    assert t.begin(visible=True, distance=2.0, region_id=(0, 0)) == 5.2
    parts = step(t, True, 1.9)
    assert "first_detection" not in parts and "new_region" not in parts
    assert abs(parts["approach"] - 0.1) < 1e-9
