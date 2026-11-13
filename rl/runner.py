"""Run proposal experiments using identical spawns, routes and safety settings."""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rl.core.agent import QLearningAgent, epsilon_schedule
from rl.core.baselines import RandomPolicy, RuleBasedTracker
from rl.core.behaviors import LostTargetSearch
from rl.core.rewards import RewardTracker
from rl.core.state import NONE, discretize, encode, next_last_seen, num_states
from rl.metrics import EpisodeMetrics


def run_episode(env, policy, cfg, rewards, episode, seed, spawn, route, epsilon=0, train=False,
                trace=None):
    obs = env.reset(spawn=spawn, route=route)
    tracker = RewardTracker(rewards)
    search = LostTargetSearch(**cfg.get("lost_search", {}))
    metrics = EpisodeMetrics(episode=episode, seed=seed, spawn=str(spawn), route=route,
                             policy="qlearning" if isinstance(policy, QLearningAgent) else type(policy).__name__)
    last_seen = NONE
    state = discretize(obs, last_seen, cfg)
    _, initial = env.observe()
    metrics.observe_pose(*initial["pose"][:2], initial["sim_s"])
    # Reset observations precede the first action. Do not credit an arbitrary
    # action with the first-detection bonus when the target is already visible.
    initial_reward = tracker.begin(visible=obs.npc_visible, distance=obs.npc_distance_m,
                                   region_id=initial["region"])
    metrics.begin(visible=obs.npc_visible, region=initial["region"], sim_s=initial["sim_s"],
                  reward=initial_reward)
    if trace:
        trace.write(json.dumps(dict(event="reset", episode=episode, step=0, action=None,
            visible=obs.npc_visible, reward=initial_reward, **initial))+"\n")
        trace.flush()
    while True:
        if isinstance(policy, QLearningAgent):
            action, _ = policy.act(encode(state, cfg), epsilon)
        else:
            action = policy.act(state)
        # A shared search routine uses only the last observed direction and lidar.
        if not obs.npc_visible and last_seen != NONE and cfg.get("lost_search_enabled", True):
            action = search.next_action(last_seen, env.observe()[1]["clearances"])
        else:
            search.reset()
        next_obs, info = env.step(action)
        reward, breakdown = tracker.step(visible=next_obs.npc_visible,
            distance=next_obs.npc_distance_m, region_id=info["region"],
            collided=info["new_collision"], captured=info["captured"])
        remembered = next_last_seen(state, last_seen)
        next_state = discretize(next_obs, remembered, cfg)
        if train:
            policy.update(encode(state, cfg), action, reward, encode(next_state, cfg), info["done"])
        metrics.observe_pose(*info["pose"][:2], info["sim_s"])
        metrics.observe_step(visible=next_obs.npc_visible, reward=reward,
            new_collision=info["new_collision"], captured=info["captured"],
            region=info["region"], sim_s=info["sim_s"])
        if trace:
            trace.write(json.dumps(dict(episode=episode, step=metrics.steps, action=action,
                visible=next_obs.npc_visible, reward=reward, rewards=breakdown, **info))+"\n")
            trace.flush()
        if metrics.steps % 20 == 0:
            print(f"episode={episode} step={metrics.steps} visible={next_obs.npc_visible} "
                  f"return={metrics.episode_return:.2f} collisions={metrics.collisions}", flush=True)
        state, obs, last_seen = next_state, next_obs, remembered
        if info["done"]:
            metrics.termination = info["termination"]
            env.stop()
            return metrics.to_dict()


def parser(training=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rl/config/qlearning.yaml")
    p.add_argument("--rewards", default="rl/config/rewards.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--route", default=None)
    p.add_argument("--spawn", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW"))
    p.add_argument("--output", default=None)
    p.add_argument("--no-exploration-bonus", action="store_true")
    p.add_argument("--no-rediscovery-bonus", action="store_true")
    p.add_argument("--no-lost-search", action="store_true")
    if training:
        p.add_argument("--resume", default=None)
        p.add_argument("--epsilon-decay-episodes", type=int, default=None)
    else:
        p.add_argument("--policy", default="rule", help="rule, random, or path to q_table.npy")
    return p


def run(args, training=False):
    import rclpy
    from rclpy.node import Node
    from rl.ros.env import ChaseEnv
    from rl.evaluate import make_report, write_report
    cfg = yaml.safe_load(Path(args.config).read_text())
    rewards = yaml.safe_load(Path(args.rewards).read_text())
    cfg["seed"] = args.seed
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    elif not training:
        cfg["episodes"] = 6
    if training and args.epsilon_decay_episodes is not None:
        cfg["epsilon"]["decay_episodes"] = args.epsilon_decay_episodes
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if cfg["episodes"] < 1 or cfg["max_steps"] < 1:
        raise ValueError("episodes and max-steps must be positive")
    if args.spawn is not None:
        cfg["spawns"] = [args.spawn]
    if args.route:
        cfg["routes"] = [args.route]
    cfg["lost_search_enabled"] = not args.no_lost_search
    rewards["new_region_enabled"] = not args.no_exploration_bonus
    rewards["rediscovery_enabled"] = not args.no_rediscovery_bonus
    rng = np.random.default_rng(args.seed)
    name = "qlearning" if training else args.policy
    output = Path(args.output or f"experiments/{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{Path(name).stem}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.yaml").write_text(yaml.safe_dump(cfg))
    (output/"rewards.yaml").write_text(yaml.safe_dump(rewards))
    if not training and args.policy in ("rule", "random"):
        policy = RuleBasedTracker() if args.policy == "rule" else RandomPolicy(rng)
    else:
        policy = QLearningAgent(num_states(cfg), cfg["alpha"], cfg["gamma"], rng)
        table = args.resume if training else args.policy
        if table:
            policy.load(table)
            if policy.q.shape != (num_states(cfg), 4):
                raise ValueError("Q-table shape does not match the current state/action configuration")
    started = time.monotonic()
    episodes = []
    rclpy.init()
    node = Node("chase_train" if training else "chase_evaluate")
    env = None
    try:
        env = ChaseEnv(node, cfg)
        routes = cfg.get("routes", ["workshop_visit", "corridor_loop"])
        with (output/"episodes.csv").open("w", newline="") as csvfile, (output/"trace.jsonl").open("w") as trace:
            writer = None
            for episode in range(cfg["episodes"]):
                spawn = cfg["spawns"][episode % len(cfg["spawns"])]
                route = routes[(episode // len(cfg["spawns"])) % len(routes)]
                epsilon = 0.0
                if training:
                    schedule = cfg["epsilon"]
                    epsilon = epsilon_schedule(episode, schedule["start"], schedule["end"],
                                               schedule["decay_episodes"])
                row = run_episode(env, policy, cfg, rewards, episode, args.seed, spawn, route,
                                  epsilon=epsilon, train=training, trace=trace)
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
                episodes.append(row)
                if training:
                    policy.save(output/"q_table.npy")
                write_report(make_report(name, episodes, started, dict(environment=cfg, rewards=rewards)),
                             output/"report.json")
                print(f"FINISHED episode={episode} termination={row['termination']} "
                      f"steps={row['steps']} return={row['episode_return']:.2f}", flush=True)
    finally:
        if env is not None:
            env.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return output
