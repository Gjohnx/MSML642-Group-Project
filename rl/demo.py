"""Explore-then-chase demo.

The robot explores with the explore Q-table until someone clicks the "Chase"
button in Gazebo, then switches to the trained chase Q-table and chases the NPC.

    python3 -m rl.demo --explore-table experiments/explore_seed0/q_table.npy \\
                       --chase-table experiments/stage2_seed0/q_table.npy

The button publishes on /demo/chase. The same signal can be sent from a terminal:

    ros2 topic pub --once /demo/chase std_msgs/msg/String "{data: chase}"

--chase-after N switches automatically after N steps (for headless runs and recordings).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rl.core.agent import ACTION_NAMES, QLearningAgent
from rl.core.baselines import RuleBasedTracker
from rl.core.explore import CHASE, EXPLORE, ExploreThenChase, num_explore_states
from rl.core.state import num_states

CHASE_TOPIC = "/demo/chase"
STATUS_EVERY_STEPS = 10


# runs the demo loop until capture, too many collisions, or the step limit
# chase_requested() returns True once the Chase button has been pressed
# get_odometry returns the robot's (x, y) from wheel odometry
# returns a summary dict, and calls log(dict) once per step if given
def run_demo(env, controller, get_odometry, chase_requested, max_steps, log=None,
             on_mode=None):
    obs = env.reset()
    controller.reset()
    if on_mode is not None:
        on_mode(EXPLORE)
    x, y = get_odometry()
    controller.observe_odometry(x, y)
    first_obs, info = env.observe()
    clear = info["clearances"]

    steps = 0
    seen_while_exploring = obs.npc_visible
    termination = "step_limit"
    while steps < max_steps:
        if controller.mode == EXPLORE and chase_requested():
            controller.request_chase(steps)
            if on_mode is not None:
                on_mode(CHASE)
            print(
                f"[demo] step {steps}: Chase pressed, switching to chase mode",
                flush=True,
            )

        action = controller.choose_action(obs, clear)
        obs, info = env.step(action)
        x, y = get_odometry()
        controller.observe_odometry(x, y)
        clear = info["clearances"]
        steps += 1
        if controller.mode == EXPLORE and obs.npc_visible:
            seen_while_exploring = True

        if log is not None:
            record = {}
            record["step"] = steps
            record["mode"] = controller.mode
            record["action"] = ACTION_NAMES[action]
            record["visible"] = obs.npc_visible
            record["pose"] = info["pose"]
            record["npc_pose"] = info["npc_pose"]
            record["new_collision"] = info["new_collision"]
            log(record)
        if steps % STATUS_EVERY_STEPS == 0:
            print(
                f"[demo] step {steps} mode={controller.mode} "
                f"npc_visible={obs.npc_visible} collisions={info['collisions']}",
                flush=True,
            )

        # touching the NPC while exploring does not count as a capture
        if info["captured"] and controller.mode == CHASE:
            termination = "capture"
            break
        if info["termination"] == "collisions":
            termination = "collisions"
            break

    env.stop()
    summary = {}
    summary["termination"] = termination
    summary["steps"] = steps
    summary["switch_step"] = controller.switch_step
    summary["chase_steps"] = None
    if controller.switch_step is not None:
        summary["chase_steps"] = steps - controller.switch_step
    summary["npc_seen_while_exploring"] = seen_while_exploring
    summary["explore_cells"] = len(controller.areas.visited)
    return summary


# remembers whether the Chase button was pressed, fed by the ROS subscription
class ChaseButton:
    def __init__(self, chase_after=None):
        self.pressed = False
        self.chase_after = chase_after
        self.checks = 0

    def receive(self, msg):
        if msg.data.strip().lower() == "chase":
            self.pressed = True

    # checked once per step by run_demo
    def requested(self):
        self.checks += 1
        if self.chase_after is not None and self.checks > self.chase_after:
            self.pressed = True
        return self.pressed


# writes one JSON line per demo step to trace.jsonl
class TraceWriter:
    def __init__(self, file):
        self.file = file

    def log(self, record):
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()


# loads the chase policy: a Q-table path, or "rule" for the rule-based tracker
def load_chase_policy(name, cfg, rng):
    if name == "rule":
        return RuleBasedTracker()
    agent = QLearningAgent(num_states(cfg), cfg["alpha"], cfg["gamma"], rng)
    agent.load(name)
    if agent.q.shape != (num_states(cfg), 4):
        raise ValueError(
            "Chase Q-table shape does not match the chase state configuration"
        )
    return agent


# command line options
def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="rl/config/qlearning.yaml")
    p.add_argument("--explore-config", default="rl/config/explore.yaml")
    p.add_argument("--explore-table", required=True)
    p.add_argument("--chase-table", required=True, help="chase q_table.npy, or 'rule'")
    p.add_argument(
        "--spawn", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW")
    )
    p.add_argument("--route", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--chase-after",
        type=int,
        default=None,
        help="press Chase automatically after N steps",
    )
    p.add_argument("--no-lost-search", action="store_true")
    p.add_argument("--output", default=None)
    return p


# loads both Q-tables, starts the ROS node and runs the demo in Gazebo
def main():
    args = parser().parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    explore_cfg = yaml.safe_load(Path(args.explore_config).read_text())
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = explore_cfg["demo_max_steps"]
    # the environment's own step limit must not end the demo early
    cfg["max_steps"] = max_steps + 1
    cfg["lost_search_enabled"] = not args.no_lost_search
    if args.spawn is not None:
        cfg["spawns"] = [args.spawn]
    if args.route:
        cfg["routes"] = [args.route]

    rng = np.random.default_rng(0)
    count = num_explore_states(cfg, explore_cfg)
    explore_agent = QLearningAgent(
        count, explore_cfg["alpha"], explore_cfg["gamma"], rng
    )
    explore_agent.load(args.explore_table)
    if explore_agent.q.shape != (count, 4):
        raise ValueError(
            "Explore Q-table shape does not match the explore state configuration"
        )
    chase_policy = load_chase_policy(args.chase_table, cfg, rng)
    controller = ExploreThenChase(explore_agent, chase_policy, cfg, explore_cfg)

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = f"experiments/{stamp}_demo"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    from rl.ros.env import ChaseEnv
    from rl.ros.odometry import OdometryReader

    rclpy.init()
    node = Node("explore_then_chase_demo")
    button = ChaseButton(args.chase_after)
    node.create_subscription(String, CHASE_TOPIC, button.receive, 10)
    mode_pub = node.create_publisher(String, "/demo/mode", 10)
    mode = "idle"

    def publish_mode(value=None):
        nonlocal mode
        if value is not None:
            mode = value
        mode_pub.publish(String(data=mode))

    node.create_timer(0.5, publish_mode)
    env = None
    started = time.monotonic()
    try:
        odometry = OdometryReader(node)
        env = ChaseEnv(node, cfg)
        print(
            f"[demo] exploring; click Chase in Gazebo (or publish on {CHASE_TOPIC})",
            flush=True,
        )
        with (output / "trace.jsonl").open("w") as trace:
            writer = TraceWriter(trace)
            summary = run_demo(
                env, controller, odometry.xy, button.requested, max_steps, writer.log,
                on_mode=publish_mode
            )
    finally:
        publish_mode("idle")
        if env is not None:
            env.close()
        node.destroy_node()
        rclpy.try_shutdown()
    summary["elapsed_wall_s"] = round(time.monotonic() - started, 3)
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
