"""Patrol the red differential-drive NPC along a configured waypoint loop."""
import argparse
import math
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from rl.core.geometry import normalization, pose_from_odom


class PatrolNode(Node):
    def __init__(self, route="workshop_visit"):
        super().__init__("npc_patrol")
        self.routes = yaml.safe_load(Path(__file__).with_name("npc_routes.yaml").read_text())["routes"]
        self.pose = None
        self.route = route
        self.index = 1
        self.enabled = True
        self.pub = self.create_publisher(Twist, "/npc/cmd_vel", 10)
        self.create_subscription(Odometry, "/npc/odom", self.receive_pose, qos_profile_sensor_data)
        self.create_subscription(String, "/npc/route", self.set_route, 10)
        self.create_timer(0.05, self.tick)

    def receive_pose(self, msg):
        self.pose = pose_from_odom(msg)

    def set_route(self, msg):
        if msg.data == "pause":
            self.enabled = False
            self.pub.publish(Twist())
        elif msg.data in self.routes:
            self.route = msg.data
            self.index = 1
            self.enabled = True

    def tick(self):
        if self.pose is None or not self.enabled:
            return
        route = self.routes[self.route]
        points = route["waypoints"]
        x, y, yaw = self.pose
        goal = points[self.index % len(points)]
        dx, dy = goal[0] - x, goal[1] - y
        if math.hypot(dx, dy) < 0.16:
            self.index += 1
            return
        error = normalization(math.atan2(dy, dx) - yaw)
        cmd = Twist()
        cmd.angular.z = max(-0.8, min(0.8, 2.0 * error))
        if abs(error) < 0.35:
            cmd.linear.x = min(route["speed_mps"], math.hypot(dx, dy))
        self.pub.publish(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="workshop_visit")
    args = parser.parse_args()
    rclpy.init()
    node = PatrolNode(args.route)
    try:
        rclpy.spin(node)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
