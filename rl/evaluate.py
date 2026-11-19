"""Evaluate a policy: `python3 -m rl.evaluate --policy {<q_table.npy>|random|rule}`.

Runs greedy episodes over the spawn/route grid and writes a JSON report with
capture rate, time to first detection, time to capture, collisions,
rediscovery rate and average episode reward.

The report carries a schema tag, the command that produced it, and both
wall and sim elapsed time, so results from different machines can be
compared on sim time.
All policies use the same Gazebo environment, sensor limits and recovery routine.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

SCHEMA = "msml642-chase-evaluation-v1"


# averages the values that are present, None when there are none
def mean_of(values):
    present = []
    for value in values:
        if value is not None:
            present.append(value)
    if len(present) == 0:
        return None
    return statistics.fmean(present)


# simulation seconds from the start of an episode to one of its events
def seconds_since_start(episode, event_key):
    event_s = episode[event_key]
    start_s = episode["first_sim_s"]
    if event_s is None or start_s is None:
        return None
    return event_s - start_s


# turns EpisodeMetrics.to_dict() rows into the proposal's metrics
def summarize(episodes):
    count = len(episodes)
    captured = []
    for episode in episodes:
        if episode["captured"]:
            captured.append(episode)

    losses = 0
    rediscoveries = 0
    for episode in episodes:
        losses += episode["losses"]
        rediscoveries += episode["rediscoveries"]

    detection_steps = []
    detection_times = []
    detected = []
    collisions = []
    returns = []
    for episode in episodes:
        detection_steps.append(episode["first_detection_step"])
        detection_times.append(seconds_since_start(episode, "first_detection_sim_s"))
        if episode["first_detection_step"] is not None:
            detected.append(1.0)
        else:
            detected.append(0.0)
        collisions.append(episode["collisions"])
        returns.append(episode["episode_return"])

    capture_steps = []
    capture_times = []
    for episode in captured:
        capture_steps.append(episode["capture_step"])
        capture_times.append(seconds_since_start(episode, "capture_sim_s"))

    capture_rate = None
    if count > 0:
        capture_rate = len(captured) / count
    rediscovery_rate = None
    if losses > 0:
        rediscovery_rate = rediscoveries / losses

    summary = {}
    summary["episodes"] = count
    summary["capture_rate"] = capture_rate
    summary["mean_steps_to_first_detection"] = mean_of(detection_steps)
    summary["mean_sim_s_to_first_detection"] = mean_of(detection_times)
    summary["mean_steps_to_capture"] = mean_of(capture_steps)
    summary["mean_sim_s_to_capture"] = mean_of(capture_times)
    summary["mean_collisions"] = mean_of(collisions)
    summary["rediscovery_rate"] = rediscovery_rate
    summary["mean_return"] = mean_of(returns)
    summary["detection_rate"] = mean_of(detected)
    summary["mean_shielded_rate"] = mean_of(
        e["shielded_steps"]/e["steps"] for e in episodes if e.get("steps") and "shielded_steps" in e)
    summary["mean_search_override_rate"] = mean_of(
        e["search_steps"]/e["steps"] for e in episodes if e.get("steps") and "search_steps" in e)
    summary["mean_recovery_events"] = mean_of(e.get("recovery_events") for e in episodes)
    return summary


# builds the report saved next to each run
def make_report(policy, episodes, started_wall, config):
    elapsed_sim = 0.0
    for episode in episodes:
        if episode["sim_elapsed_s"]:
            elapsed_sim += episode["sim_elapsed_s"]

    report = {}
    report["schema"] = SCHEMA
    report["policy"] = policy
    report["command"] = " ".join(sys.argv)
    report["elapsed_wall_s"] = round(time.monotonic() - started_wall, 3)
    report["elapsed_sim_s"] = round(elapsed_sim, 3)
    report["config"] = config
    report["summary"] = summarize(episodes)
    report["episodes"] = episodes
    return report


# writes to a temp file first, then replaces the old report
def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, path)


def main():
    from rl.runner import parser, run

    run(parser().parse_args())


if __name__ == "__main__":
    main()
