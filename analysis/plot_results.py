"""Plot saved Gazebo learning curves and measured policy comparisons."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="Experiment directories with report.json and episodes.csv")
    p.add_argument("--output", default="docs/figures")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for run in map(Path, args.runs):
        report = json.loads((run/"report.json").read_text())
        reports.append((run.name, report))
        rows = list(csv.DictReader((run/"episodes.csv").open()))
        if (run/"q_table.npy").exists():
            returns = np.array([float(row["episode_return"]) for row in rows])
            window = min(10, len(returns))
            axes[0].plot(np.arange(window, len(returns)+1),
                         np.convolve(returns, np.ones(window)/window, mode="valid"),
                         label=run.name)
    axes[0].set(xlabel="Training episode", ylabel="Episode return (moving average)")
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    names = [name for name, _ in reports]
    rates = [r["summary"]["capture_rate"] or 0 for _, r in reports]
    axes[1].bar(np.arange(len(names)), rates, color="#3478a5")
    axes[1].set(xticks=np.arange(len(names)), xticklabels=names, ylabel="Observed capture rate", ylim=(0, 1))
    axes[1].tick_params(axis="x", rotation=35, labelsize=7)
    fig.savefig(out/"learning_and_capture.png", dpi=180)
    plt.close(fig)
    fields = ["episodes", "capture_rate", "mean_sim_s_to_first_detection",
              "mean_sim_s_to_capture", "mean_collisions", "rediscovery_rate", "mean_return"]
    with (out/"comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", *fields])
        w.writeheader()
        for name, report in reports:
            w.writerow(dict(run=name, **{k: report["summary"][k] for k in fields}))


if __name__ == "__main__":
    main()
