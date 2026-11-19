"""Regression checks for visibility and capture across static obstacles."""
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

from rl.core.world import line_of_sight, load_obstacles
from rl.ros.env import ChaseEnv
import subprocess


def test_visibility_handles_thin_rotated_boxes_and_cylinders():
    assert not line_of_sight((-1, 0), (1, 0), [("box", 0, 0, .4, .001, 1)])
    assert not line_of_sight((-1, 0), (1, 0), [("cylinder", 0, 0, 0, .2, 0)])
    assert line_of_sight((-1, 1), (1, 1), [("cylinder", 0, 0, 0, .2, 0)])


def test_capture_cannot_cross_north_partition():
    cfg = yaml.safe_load(Path("rl/config/qlearning.yaml").read_text())
    scan = SimpleNamespace(angle_min=-math.pi, angle_increment=math.pi/180,
        range_min=.05, range_max=6., ranges=[.05]*360)
    env = SimpleNamespace(scan=scan, pose=(0, 2.12, math.pi/2), npc_pose=(0, 2.60, 0),
        cfg=cfg, collisions=SimpleNamespace(count=0), sim_s=0,
        obstacles=load_obstacles("sim/room.sdf"))
    obs, info = ChaseEnv.observe(env)
    assert math.dist(env.pose[:2], env.npc_pose[:2]) < cfg["capture_distance_m"]
    assert not obs.npc_visible and not info["captured"]
    env.npc_pose = (0, 1.8, 0)
    assert ChaseEnv.observe(env)[1]["captured"]


def test_pose_reset_retries_a_transient_timeout(monkeypatch):
    results = iter([subprocess.CompletedProcess([], 0, "", "Service call timed out"),
                    subprocess.CompletedProcess([], 0, "data: true", "")])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return next(results)
    monkeypatch.setattr(subprocess, "run", run)
    ChaseEnv.set_pose(SimpleNamespace(), "robot", (0, 0, 0))
    assert len(calls) == 2
