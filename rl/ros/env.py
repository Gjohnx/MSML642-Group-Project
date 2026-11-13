"""Gazebo episode interface. The policy receives only local sensor observations."""
import math
import subprocess
import time
from pathlib import Path

import yaml

from rl.core.state import Observation


MAX_SENSOR_AGE_S = 2.0


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def external_contacts(collision_pairs, robot_model, ignore=("floor",)):
    prefixes = (f"{robot_model}::",) + tuple(f"{name}::" for name in ignore)
    return list(dict.fromkeys(name for pair in collision_pairs for name in pair
                              if not name.startswith(prefixes)))


class SensorFreshness:
    """Wall-clock arrival barrier for subscriptions after a motion command."""
    def __init__(self, names=("odom", "scan"), max_age_s=MAX_SENSOR_AGE_S, clock=time.monotonic):
        self.names = tuple(names)
        self.max_age_s = max_age_s
        self.clock = clock
        self.last = {name: None for name in self.names}
        self.after = 0.0

    def mark(self, name):
        self.last[name] = self.clock()

    def barrier(self):
        self.after = self.clock()

    def ready(self):
        now = self.clock()
        return all(t is not None and self.after <= t <= now and now - t <= self.max_age_s
                   for t in self.last.values())

    def ages(self):
        now = self.clock()
        return {k: None if t is None else round(now-t, 3) for k, t in self.last.items()}


class ChaseEnv:
    """Drive in simulation time and reset both physical robots between episodes."""
    def __init__(self, node, cfg):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from ros_gz_interfaces.msg import Contacts
        from std_msgs.msg import String
        from rclpy.qos import qos_profile_sensor_data
        from rl.core.behaviors import CollisionCounter, CollisionRecovery
        self.node, self.cfg = node, cfg
        self.rclpy, self.Twist, self.String = rclpy, Twist, String
        self.pose = self.npc_pose = self.scan = None
        self.sim_s = 0.0
        self.scan_s = self.npc_s = 0.0
        self.contact = False
        self.fresh = SensorFreshness(("odom", "scan", "npc"), max_age_s=10)
        self.velocity = node.create_publisher(Twist, "/robot/cmd_vel", 10)
        self.npc_velocity = node.create_publisher(Twist, "/npc/cmd_vel", 10)
        self.route_pub = node.create_publisher(String, "/npc/route", 10)
        node.create_subscription(Odometry, "/robot/ground_truth", self.receive_pose, qos_profile_sensor_data)
        node.create_subscription(Odometry, "/npc/odom", self.receive_npc, qos_profile_sensor_data)
        node.create_subscription(LaserScan, "/robot/scan", self.receive_scan, qos_profile_sensor_data)
        node.create_subscription(Contacts, "/robot/contact", self.receive_contact, qos_profile_sensor_data)
        self.collisions = CollisionCounter(cfg["max_collisions"])
        self.recovery = CollisionRecovery(**cfg.get("recovery", {}))
        self.routes = yaml.safe_load((Path(__file__).parents[2]/"sim/npc_routes.yaml").read_text())["routes"]
        self.wait(lambda: self.fresh.ready() and self.velocity.get_subscription_count() > 0
                  and self.route_pub.get_subscription_count() > 0, 90, "ROS sensors or NPC patrol unavailable")
        self.steps = 0

    def receive_pose(self, msg):
        from rl.core.geometry import pose_from_odom
        self.pose = pose_from_odom(msg)
        self.sim_s = stamp_seconds(msg.header.stamp)
        self.fresh.mark("odom")

    def receive_npc(self, msg):
        from rl.core.geometry import pose_from_odom
        self.npc_pose = pose_from_odom(msg)
        self.npc_s = stamp_seconds(msg.header.stamp)
        self.fresh.mark("npc")

    def receive_scan(self, msg):
        self.scan = msg
        self.scan_s = stamp_seconds(msg.header.stamp)
        self.fresh.mark("scan")

    def receive_contact(self, msg):
        pairs = [(c.collision1.name, c.collision2.name) for c in msg.contacts]
        # Contact pulses are latched until consumed at the next step.
        self.contact = self.contact or bool(external_contacts(pairs, "robot", ("floor", "npc")))

    def wait(self, condition, timeout, message):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"{message}. Sensor ages: {self.fresh.ages()}")
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def stop(self):
        self.velocity.publish(self.Twist())

    def drive(self, linear, angular, seconds):
        cmd = self.Twist()
        cmd.linear.x, cmd.angular.z = float(linear), float(angular)
        finish = self.sim_s + seconds
        deadline = time.monotonic() + max(30, seconds * 30)
        try:
            while self.sim_s < finish:
                if time.monotonic() > deadline:
                    raise RuntimeError("Simulation clock stalled during action")
                self.velocity.publish(cmd)
                self.rclpy.spin_once(self.node, timeout_sec=0.01)
        finally:
            self.stop()
        self.fresh.barrier()
        self.wait(lambda: self.fresh.ready() and self.scan_s >= finish and self.npc_s >= finish,
                  30, "No fresh observations after action")

    def set_pose(self, name, pose):
        x, y, yaw = pose
        request = (f'name: "{name}", position: {{x: {x}, y: {y}, z: 0.005}}, '
                   f'orientation: {{z: {math.sin(yaw/2)}, w: {math.cos(yaw/2)}}}')
        result = subprocess.run(["gz", "service", "-s", "/world/room/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000", "--req", request], capture_output=True, text=True, timeout=10)
        if result.returncode or "true" not in result.stdout:
            raise RuntimeError(f"Reset failed for {name}: {result.stdout} {result.stderr}")

    def reset(self, spawn=None, route=None):
        spawn = self.cfg["spawns"][0] if spawn is None else spawn
        route = route or self.cfg.get("routes", ["workshop_visit"])[0]
        points = self.routes[route]["waypoints"]
        npc = [*points[0], math.atan2(points[1][1]-points[0][1], points[1][0]-points[0][0])]
        self.route_pub.publish(self.String(data="pause"))
        self.drive(0, 0, 0.2)
        self.npc_velocity.publish(self.Twist())
        self.set_pose("robot", spawn)
        self.set_pose("npc", npc)
        self.wait(lambda: math.dist(self.pose[:2], spawn[:2]) < 0.15
                  and math.dist(self.npc_pose[:2], npc[:2]) < 0.2,
                  15, "Reset poses did not arrive on the ROS bridge")
        self.drive(0, 0, 0.3)
        if math.dist(self.pose[:2], spawn[:2]) > 0.15 or math.dist(self.npc_pose[:2], npc[:2]) > 0.2:
            raise RuntimeError("Robot reset did not reach the requested world poses")
        self.contact = False
        self.collisions.reset()
        self.steps = 0
        self.route_pub.publish(self.String(data=route))
        return self.observe()[0]

    def observe(self):
        from rl.core.geometry import scan_to_rays, clearances
        from rl.ros.detector import detect
        rays = scan_to_rays(self.scan)
        clear = clearances(rays, self.scan.range_max)
        visible, bearing, distance = detect(self.pose, self.npc_pose[:2], rays, **self.cfg["detector"])
        obs = Observation(visible, bearing, distance, clear["front"], clear["left"], clear["right"])
        captured = math.dist(self.pose[:2], self.npc_pose[:2]) <= self.cfg["capture_distance_m"]
        cell = self.cfg.get("exploration_cell_m", 2)
        info = dict(pose=self.pose, npc_pose=self.npc_pose, sim_s=self.sim_s,
                    region=(math.floor((self.pose[0]+8)/cell), math.floor((self.pose[1]+6)/cell)),
                    captured=captured, collisions=self.collisions.count, clearances=clear)
        return obs, info

    def step(self, action):
        from rl.core.agent import ACTION_NAMES
        from rl.core.behaviors import SafetyConfig, motion_unsafe
        from rl.core.geometry import scan_to_rays
        if action not in range(4):
            raise ValueError("action must be forward, left, right or stop (0..3)")
        cmd = self.cfg["actions"][ACTION_NAMES[action]]
        shielded = cmd["linear"] > 0 and motion_unsafe(scan_to_rays(self.scan),
            self.scan.range_max, 0.0, SafetyConfig(**self.cfg.get("safety", {})))
        self.drive(0 if shielded else cmd["linear"], cmd["angular"], self.cfg["action_duration_s"])
        new_collision = self.collisions.observe(self.contact)
        self.contact = False
        obs, info = self.observe()
        if new_collision and not info["captured"] and not self.collisions.done:
            for command in self.recovery.plan(info["clearances"]):
                self.drive(command.linear, command.angular, command.duration_s)
            obs, info = self.observe()
        self.steps += 1
        info.update(new_collision=new_collision, shielded=shielded, steps=self.steps)
        info["termination"] = ("capture" if info["captured"] else "collisions" if self.collisions.done
                               else "step_limit" if self.steps >= self.cfg["max_steps"] else "")
        info["done"] = bool(info["termination"])
        return obs, info

    def close(self):
        self.stop()
        self.route_pub.publish(self.String(data="pause"))
        self.npc_velocity.publish(self.Twist())
