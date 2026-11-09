"""Exercise the first-stage robot through ROS 2 and check its Gazebo scan."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec / 1e9


def heading(odometry):
    q = odometry.pose.pose.orientation
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2)
    )


def heading_change(before, after):
    difference = heading(after) - heading(before)
    return math.atan2(math.sin(difference), math.cos(difference))


class RobotCheck(Node):
    def __init__(self):
        super().__init__("robot_check")
        self.odom = None
        self.scan = None
        self.velocity = self.create_publisher(Twist, "/robot/cmd_vel", 10)
        self.create_subscription(
            Odometry, "/robot/odom", self.receive_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, "/robot/scan", self.receive_scan, qos_profile_sensor_data
        )

    def receive_odom(self, message):
        self.odom = message

    def receive_scan(self, message):
        self.scan = message

    def wait_until(self, condition, timeout, failure):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                raise RuntimeError(failure)
            rclpy.spin_once(self, timeout_sec=0.1)

    def front_range(self):
        middle = round(-self.scan.angle_min / self.scan.angle_increment)
        values = self.scan.ranges[middle - 4 : middle + 5]
        return min(values, default=math.inf)

    def drive(self, linear, angular, seconds):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        finish = stamp(self.odom) + seconds
        deadline = time.monotonic() + 30
        while stamp(self.odom) < finish:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for simulation odometry.")
            self.velocity.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.velocity.publish(Twist())
        finish = stamp(self.odom)
        self.wait_until(
            lambda: stamp(self.scan) > finish,
            10,
            "No fresh lidar scan after the command.",
        )

    def check(self):
        self.wait_until(
            lambda: self.odom is not None
            and self.scan is not None
            and self.velocity.get_subscription_count() > 0,
            60,
            "Missing ROS bridge, lidar, or odometry.",
        )
        start = self.odom
        initial_range = self.front_range()
        if not 1.6 < initial_range < 2.1:
            raise RuntimeError(f"Expected a wall near 1.8 m, got {initial_range:.2f} m.")

        self.drive(0.25, 0.0, 1.5)
        forward = self.odom.pose.pose.position.x - start.pose.pose.position.x
        final_range = self.front_range()
        if forward < 0.2 or not initial_range - final_range > 0.15:
            raise RuntimeError(
                f"Forward command failed: odom dx={forward:.2f} m, "
                f"lidar={initial_range:.2f} -> {final_range:.2f} m."
            )

        before_left = self.odom
        self.drive(0.0, 0.6, 1.0)
        left = heading_change(before_left, self.odom)
        before_right = self.odom
        self.drive(0.0, -0.6, 1.0)
        right = heading_change(before_right, self.odom)
        if left < 0.3 or right > -0.3:
            raise RuntimeError(f"Turns failed: left={left:.2f}, right={right:.2f} rad.")

        self.drive(0.0, 0.0, 0.5)
        stopped = self.odom.twist.twist
        if abs(stopped.linear.x) > 0.03 or abs(stopped.angular.z) > 0.05:
            raise RuntimeError("Robot did not stop.")

        print(
            f"PASS: forward {forward:.2f} m, wall {initial_range:.2f} -> "
            f"{final_range:.2f} m, left {left:.2f} rad, right {right:.2f} rad, stopped."
        )


def main():
    rclpy.init()
    node = RobotCheck()
    try:
        node.check()
    except RuntimeError as error:
        print(f"FAIL: {error}")
        return 1
    finally:
        node.velocity.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
