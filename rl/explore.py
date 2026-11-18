"""Train or evaluate the exploration Q-table for the explore-then-chase demo.

    python3 -m rl.explore --episodes 150 --output experiments/explore_seed0
    python3 -m rl.explore --evaluate experiments/explore_seed0/q_table.npy --episodes 6

Uses the same Gazebo environment as the chase experiments. The explore state
only has the lidar obstacle levels and the stall level from wheel odometry.
Writes q_table.npy (training only), episodes.csv and report.json.
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rl.core.agent import QLearningAgent, epsilon_schedule
from rl.core.explore import (
    ExploreRewardTracker,
    OdometryAreas,
    encode_explore,
    explore_state,
    num_explore_states,
)
from rl.evaluate import mean_of, write_report

SCHEMA = "msml642-explore-v1"


# runs one exploration episode and returns its row for episodes.csv
# env is ChaseEnv (or a fake with the same reset/observe/step/stop)
# get_odometry returns the robot's (x, y) from wheel odometry
def run_explore_episode(
    env,
    agent,
    cfg,
    explore_cfg,
    get_odometry,
    spawn,
    route,
    epsilon=0.0,
    train=False,
):
    obs = env.reset(spawn=spawn, route=route)
    areas = OdometryAreas(explore_cfg["odometry_cell_m"])
    tracker = ExploreRewardTracker(explore_cfg["rewards"])

    x, y = get_odometry()
    areas.observe(x, y)
    first_obs, info = env.observe()
    tracker.begin(info["region"])
    state = explore_state(obs, areas.steps_since_new, cfg, explore_cfg)

    steps = 0
    episode_return = 0.0
    collisions = 0
    npc_seen = obs.npc_visible
    termination = ""
    while True:
        row = encode_explore(state, cfg, explore_cfg)
        if train:
            action, explored = agent.act(row, epsilon)
        else:
            action = agent.greedy(row)
        next_obs, info = env.step(action)
        x, y = get_odometry()
        areas.observe(x, y)
        reward, parts = tracker.step(info["region"], info["new_collision"])
        next_state = explore_state(next_obs, areas.steps_since_new, cfg, explore_cfg)

        # bumping into the NPC is not a goal while exploring, only crashes and time end it
        done = info["termination"] in ("collisions", "step_limit")
        if train:
            next_row = encode_explore(next_state, cfg, explore_cfg)
            agent.update(row, action, reward, next_row, done)

        steps += 1
        episode_return += reward
        if info["new_collision"]:
            collisions += 1
        if next_obs.npc_visible:
            npc_seen = True
        state = next_state
        obs = next_obs
        if done:
            termination = info["termination"]
            break

    env.stop()
    result = {}
    result["steps"] = steps
    result["episode_return"] = episode_return
    result["regions_visited"] = len(tracker.visited_regions)
    result["odometry_cells"] = len(areas.visited)
    result["collisions"] = collisions
    result["npc_seen"] = npc_seen
    result["termination"] = termination
    return result


# averages the episode rows for report.json
def summarize_explore(episodes):
    regions = []
    cells = []
    collisions = []
    returns = []
    seen = []
    for episode in episodes:
        regions.append(episode["regions_visited"])
        cells.append(episode["odometry_cells"])
        collisions.append(episode["collisions"])
        returns.append(episode["episode_return"])
        if episode["npc_seen"]:
            seen.append(1.0)
        else:
            seen.append(0.0)
    summary = {}
    summary["episodes"] = len(episodes)
    summary["mean_regions_visited"] = mean_of(regions)
    summary["mean_odometry_cells"] = mean_of(cells)
    summary["mean_collisions"] = mean_of(collisions)
    summary["mean_return"] = mean_of(returns)
    summary["npc_seen_rate"] = mean_of(seen)
    return summary


# command line options
def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="rl/config/qlearning.yaml")
    p.add_argument("--explore-config", default="rl/config/explore.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", default=None, help="continue training from this Q-table")
    p.add_argument(
        "--evaluate", default=None, help="run this Q-table greedily, no training"
    )
    p.add_argument("--output", default=None)
    return p


# loads the configs and the Q-table, then runs the episodes in Gazebo
def main():
    args = parser().parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    explore_cfg = yaml.safe_load(Path(args.explore_config).read_text())
    training = args.evaluate is None

    if args.episodes is not None:
        explore_cfg["episodes"] = args.episodes
    elif not training:
        explore_cfg["episodes"] = 6
    if args.max_steps is not None:
        explore_cfg["max_steps"] = args.max_steps
    if explore_cfg["episodes"] < 1 or explore_cfg["max_steps"] < 1:
        raise ValueError("episodes and max-steps must be positive")
    # the environment ends episodes at cfg["max_steps"]
    cfg["max_steps"] = explore_cfg["max_steps"]

    rng = np.random.default_rng(args.seed)
    count = num_explore_states(cfg, explore_cfg)
    agent = QLearningAgent(count, explore_cfg["alpha"], explore_cfg["gamma"], rng)
    table = args.resume
    if not training:
        table = args.evaluate
    if table:
        agent.load(table)
        if agent.q.shape != (count, 4):
            raise ValueError(
                "Q-table shape does not match the explore state configuration"
            )

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        kind = "train"
        if not training:
            kind = "eval"
        output = f"experiments/{stamp}_explore_{kind}_seed{args.seed}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg))
    (output / "explore.yaml").write_text(yaml.safe_dump(explore_cfg))

    import rclpy
    from rclpy.node import Node

    from rl.ros.env import ChaseEnv
    from rl.ros.odometry import OdometryReader

    started = time.monotonic()
    episodes = []
    rclpy.init()
    node_name = "explore_train"
    if not training:
        node_name = "explore_evaluate"
    node = Node(node_name)
    env = None
    try:
        odometry = OdometryReader(node)
        env = ChaseEnv(node, cfg)
        routes = cfg.get("routes", ["workshop_visit"])
        spawns = cfg["spawns"]
        with (output / "episodes.csv").open("w", newline="") as csvfile:
            writer = None
            for episode in range(explore_cfg["episodes"]):
                spawn = spawns[episode % len(spawns)]
                route = routes[(episode // len(spawns)) % len(routes)]
                epsilon = 0.0
                if training:
                    schedule = explore_cfg["epsilon"]
                    epsilon = epsilon_schedule(
                        episode,
                        schedule["start"],
                        schedule["end"],
                        schedule["decay_episodes"],
                    )
                row = run_explore_episode(
                    env,
                    agent,
                    cfg,
                    explore_cfg,
                    odometry.xy,
                    spawn,
                    route,
                    epsilon,
                    training,
                )
                row["episode"] = episode
                row["seed"] = args.seed
                row["spawn"] = str(spawn)
                row["route"] = route
                row["epsilon"] = epsilon
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
                episodes.append(row)
                if training:
                    agent.save(output / "q_table.npy")

                report = {}
                report["schema"] = SCHEMA
                report["policy"] = "explore_qlearning"
                if not training:
                    report["policy"] = str(args.evaluate)
                report["elapsed_wall_s"] = round(time.monotonic() - started, 3)
                report["config"] = {"environment": cfg, "explore": explore_cfg}
                report["summary"] = summarize_explore(episodes)
                report["episodes"] = episodes
                write_report(report, output / "report.json")
                print(
                    f"FINISHED episode={episode} regions={row['regions_visited']} "
                    f"collisions={row['collisions']} return={row['episode_return']:.2f}",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(summarize_explore(episodes), indent=1))
    return output


if __name__ == "__main__":
    main()
