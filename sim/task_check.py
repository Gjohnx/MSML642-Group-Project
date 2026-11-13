"""Check reset, limited-FOV sensing, capture and contact recovery in running Gazebo."""
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node

from rl.core.agent import FORWARD, STOP
from rl.ros.env import ChaseEnv


def main():
    cfg = yaml.safe_load((Path(__file__).parents[1]/"rl/config/qlearning.yaml").read_text())
    rclpy.init()
    node = Node("task_integration_check")
    env = None
    try:
        env = ChaseEnv(node, cfg)
        obs = env.reset(spawn=[-3, 1.6, 0], route="workshop_visit")
        assert obs.npc_visible and 1.7 < obs.npc_distance_m < 2.3, "Visible spawn detection failed"
        obs = env.reset(spawn=[0, 0, 0], route="workshop_visit")
        assert not obs.npc_visible and obs.npc_distance_m is None, "Hidden target leaked into observation"
        env.route_pub.publish(env.String(data="pause"))
        env.drive(0, 0, 0.2)
        env.set_pose("npc", [3, 0, 0])
        env.drive(0, 0, 0.6)
        assert not env.observe()[0].npc_visible, "Target behind aisle barrier should be occluded"
        obs = env.reset(spawn=[-1.56, 1.6, 0], route="workshop_visit")
        captured = False
        for _ in range(12):
            obs, info = env.step(FORWARD)
            if info["captured"]:
                captured = True
                break
        assert captured and info["termination"] == "capture", "Capture termination failed"
        env.reset(spawn=[0, 0, 0], route="workshop_visit")
        # Deliberately bypass the forward shield to test an actual body contact.
        env.drive(0.25, 0, 8)
        assert env.contact, "Body contact sensor did not report the aisle barrier"
        _, info = env.step(STOP)
        assert info["new_collision"] and info["collisions"] == 1, "Contact was not counted/recovered"
        print("PASS: visible/hidden/occluded target, reset, moving-target capture, contact and recovery")
    finally:
        if env:
            env.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
