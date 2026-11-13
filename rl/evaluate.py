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


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summarize(episodes: list[dict]) -> dict:
    """Aggregate EpisodeMetrics.to_dict() rows into the proposal's metrics."""
    n = len(episodes)
    captured = [e for e in episodes if e["captured"]]
    losses = sum(e["losses"] for e in episodes)
    return {
        "episodes": n,
        "capture_rate": len(captured) / n if n else None,
        "mean_steps_to_first_detection": _mean(e["first_detection_step"] for e in episodes),
        "mean_sim_s_to_first_detection": _mean(
            (e["first_detection_sim_s"] - e["first_sim_s"])
            if e["first_detection_sim_s"] is not None and e["first_sim_s"] is not None else None
            for e in episodes),
        "mean_steps_to_capture": _mean(e["capture_step"] for e in captured),
        "mean_sim_s_to_capture": _mean(
            (e["capture_sim_s"] - e["first_sim_s"])
            if e["capture_sim_s"] is not None and e["first_sim_s"] is not None else None
            for e in captured),
        "mean_collisions": _mean(e["collisions"] for e in episodes),
        "rediscovery_rate": (sum(e["rediscoveries"] for e in episodes) / losses) if losses else None,
        "mean_return": _mean(e["episode_return"] for e in episodes),
        "detection_rate": _mean(1.0 if e["first_detection_step"] is not None else 0.0
                                for e in episodes),
    }


def make_report(policy: str, episodes: list[dict], started_wall: float, config: dict) -> dict:
    return {
        "schema": SCHEMA,
        "policy": policy,
        "command": " ".join(sys.argv),
        "elapsed_wall_s": round(time.monotonic() - started_wall, 3),
        "elapsed_sim_s": round(sum(e["sim_elapsed_s"] or 0.0 for e in episodes), 3),
        "config": config,
        "summary": summarize(episodes),
        "episodes": episodes,
    }


def write_report(report: dict, path) -> None:
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
