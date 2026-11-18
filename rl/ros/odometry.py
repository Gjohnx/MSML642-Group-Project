"""Keeps the latest wheel odometry (/robot/odom), the robot's own position estimate.

This is not the true position: the explore state may use it, the true pose may not.
"""

from rl.core.geometry import pose_from_odom


# subscribes to /robot/odom on an existing node and remembers the latest pose
class OdometryReader:
    def __init__(self, node, topic="/robot/odom"):
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data

        self.pose = None
        node.create_subscription(Odometry, topic, self.receive, qos_profile_sensor_data)

    def receive(self, msg):
        self.pose = pose_from_odom(msg)

    # returns (x, y) of the latest odometry pose
    def xy(self):
        if self.pose is None:
            raise RuntimeError("No /robot/odom message received yet")
        return (self.pose[0], self.pose[1])
