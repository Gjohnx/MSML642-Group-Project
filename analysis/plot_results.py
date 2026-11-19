"""Plot learning curves and policy comparisons from saved experiment runs.

    python3 -m analysis.plot_results experiments/suite/* --output docs/figures

Each run directory needs a report.json. Runs that differ only in their
_seed<N> suffix are treated as repeats of one policy and averaged.
Writes learning_curves.png, policy_comparison.png, summary.csv (mean and
spread per policy) and comparison.csv (one row per run).
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# one fixed color per policy, so a policy keeps its color in every figure
POLICY_COLORS = {
    "full": "royalblue",
    "no_exploration": "chocolate",
    "no_rediscovery": "lightseagreen",
    "random": "goldenrod",
    "rule": "palevioletred",
}
EXTRA_COLORS = ["green", "slateblue", "indianred"]

# names shown on the figures
POLICY_NAMES = {
    "full": "Q-learning, full rewards",
    "no_exploration": "Q-learning, no exploration bonus",
    "no_rediscovery": "Q-learning, no rediscovery bonus",
    "random": "Random",
    "rule": "Rule-based",
}
HELDOUT_SUFFIX = "_heldout"

# figure colors
SURFACE_COLOR = "white"
TEXT_COLOR = "black"
LABEL_COLOR = "dimgray"
NOTE_COLOR = "gray"
GRID_COLOR = "gainsboro"
AXIS_COLOR = "silver"

# figure sizes
BAR_HEIGHT = 0.56
LINE_WIDTH = 2
MAX_SMOOTH_WINDOW = 25
FIGURE_DPI = 180

# metrics to compare: (summary key, panel title, value format, fixed axis limit)
METRICS = [
    ("capture_rate", "Capture rate", "{:.0%}", 1.0),
    (
        "mean_sim_s_to_first_detection",
        "Time to first detection (sim s)",
        "{:.1f}",
        None,
    ),
    ("mean_sim_s_to_capture", "Time to capture (sim s)", "{:.1f}", None),
    ("mean_collisions", "Collisions per episode", "{:.2f}", None),
    ("rediscovery_rate", "Rediscovery rate", "{:.0%}", 1.0),
    ("mean_return", "Episode return", "{:.1f}", None),
]


# turns a run folder name into a policy key, train_full_seed0 -> full
def policy_key(run_name):
    name = re.sub(r"_seed\d+$", "", run_name)
    name = re.sub(r"^(train|eval)_", "", name)
    return name


# removes the held-out marker, rule_heldout -> rule
def base_policy(key):
    if key.endswith(HELDOUT_SUFFIX):
        return key[: -len(HELDOUT_SUFFIX)]
    return key


# returns the name shown on the figure for a policy key
def display_name(key):
    base = base_policy(key)
    name = POLICY_NAMES.get(base)
    if name is None:
        name = base.replace("_", " ")
    if key.endswith(HELDOUT_SUFFIX):
        name = name + " (held-out starts)"
    return name


# returns the color of a policy, unknown policies take the next extra color
def color_for(key, extras):
    base = base_policy(key)
    if base in POLICY_COLORS:
        return POLICY_COLORS[base]
    if base not in extras:
        if len(extras) < len(EXTRA_COLORS):
            extras[base] = EXTRA_COLORS[len(extras)]
        else:
            extras[base] = NOTE_COLOR
    return extras[base]


# shared look for every panel
def style_axes(ax):
    ax.set_facecolor(SURFACE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=LABEL_COLOR, labelsize=8, length=0)
    ax.title.set_color(TEXT_COLOR)
    ax.set_axisbelow(True)


def moving_average(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# reads every report.json and splits the runs into training and evaluation
def load_runs(run_dirs):
    training = {}
    evaluation = {}
    rows = []
    for run_dir in run_dirs:
        run = Path(run_dir)
        path = run / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        key = policy_key(run.name)
        if report.get("policy") == "qlearning":
            group = training
        else:
            group = evaluation
        if key not in group:
            group[key] = []
        group[key].append(report)
        rows.append((run.name, report))
    return training, evaluation, rows


# returns one array per seed for an episode field, all cut to the same length
def episode_series(reports, field):
    series = []
    for report in reports:
        values = []
        for episode in report["episodes"]:
            values.append(float(episode[field]))
        series.append(np.array(values))
    length = len(series[0])
    for values in series:
        if len(values) < length:
            length = len(values)
    trimmed = []
    for values in series:
        trimmed.append(values[:length])
    return trimmed, length


# draws return and capture rate over training, one line per policy
def plot_learning_curves(training, out, extras):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.8), layout="constrained", facecolor=SURFACE_COLOR
    )
    panels = [("episode_return", "Episode return"), ("captured", "Capture rate")]
    for index in range(len(panels)):
        field, title = panels[index]
        ax = axes[index]
        style_axes(ax)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
        for key in training:
            series, length = episode_series(training[key], field)
            if length == 0:
                continue
            window = min(MAX_SMOOTH_WINDOW, length // 4)
            if window < 1:
                window = 1
            smoothed = []
            for values in series:
                smoothed.append(moving_average(values, window))
            smoothed = np.array(smoothed)
            x = np.arange(window, length + 1)
            color = color_for(key, extras)
            ax.plot(
                x,
                smoothed.mean(axis=0),
                color=color,
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=display_name(key),
            )
            if len(series) > 1:
                lowest = smoothed.min(axis=0)
                highest = smoothed.max(axis=0)
                ax.fill_between(
                    x, lowest, highest, color=color, alpha=0.12, linewidth=0
                )
        ax.set_title(f"{title} during training", fontsize=10, loc="left")
        ax.set_xlabel("Training episode", fontsize=8, color=LABEL_COLOR)
        if field == "captured":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            ncols=min(3, len(handles)),
            frameon=False,
            fontsize=8,
            labelcolor=LABEL_COLOR,
        )
    note = "Moving average per run; line = mean across seeds, band = range across seeds"
    fig.supxlabel(note, fontsize=7, color=NOTE_COLOR)
    fig.savefig(out / "learning_curves.png", dpi=FIGURE_DPI, facecolor=SURFACE_COLOR,
                bbox_inches="tight")
    plt.close(fig)


# averages every metric across the seeds of a policy, missing values are skipped
def summarize(evaluation):
    table = {}
    for key in evaluation:
        table[key] = {}
        for metric, title, fmt, limit in METRICS:
            values = []
            for report in evaluation[key]:
                value = report["summary"].get(metric)
                if value is not None:
                    values.append(value)
            if len(values) == 0:
                table[key][metric] = (None, None, 0)
            else:
                mean = float(np.mean(values))
                std = float(np.std(values))
                table[key][metric] = (mean, std, len(values))
    return table


# largest number of seeds behind any value in the table
def seed_count(table):
    seeds = 0
    for key in table:
        for metric in table[key]:
            count = table[key][metric][2]
            if count > seeds:
                seeds = count
    return seeds


# draws one bar panel per metric, one bar per policy
def plot_policy_comparison(table, out, extras):
    keys = list(table)
    names = []
    for key in keys:
        names.append(display_name(key))
    height = 1.6 + 0.42 * len(keys) * 2
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, height),
        layout="constrained",
        facecolor=SURFACE_COLOR,
        sharey=True,
    )
    positions = np.arange(len(keys))[::-1]
    panels = axes.flat
    for index in range(len(METRICS)):
        metric, title, fmt, limit = METRICS[index]
        ax = panels[index]
        style_axes(ax)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=1)

        present = []
        for key in keys:
            mean = table[key][metric][0]
            if mean is not None:
                present.append(mean)
        low = 0.0
        high = 1.0
        if len(present) > 0:
            low = min(0.0, min(present))
            high = max(present)
        if limit is not None:
            high = limit
        # every value is zero, keep a readable axis
        if high - low < 1e-6:
            high = low + 1.0
        span = high - low

        for row in range(len(keys)):
            key = keys[row]
            y = positions[row]
            mean, std, count = table[key][metric]
            if mean is None:
                ax.text(low, y, "  no data", va="center", fontsize=7, color=NOTE_COLOR)
                continue
            ax.barh(
                y, mean, height=BAR_HEIGHT, color=color_for(key, extras), linewidth=0
            )
            spread = 0.0
            if count > 1 and std:
                spread = std
                ax.plot(
                    [mean - std, mean + std], [y, y], color=LABEL_COLOR, linewidth=1
                )
            label_x = max(mean + spread, 0) + 0.02 * span
            ax.text(
                label_x, y, fmt.format(mean), va="center", fontsize=7, color=LABEL_COLOR
            )

        left = low
        if low < 0:
            left = low - 0.05 * span
        ax.axvline(0, color=AXIS_COLOR, linewidth=1)
        ax.set_xlim(left, high + 0.18 * span)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks(positions, names, fontsize=8)
        if limit is not None:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    fig.suptitle(
        "Evaluation by policy", fontsize=11, color=TEXT_COLOR, x=0.01, ha="left"
    )
    seeds = seed_count(table)
    note = f"Bars = mean across {seeds} seed(s); thin line = ±1 standard deviation"
    fig.supxlabel(note, fontsize=7, color=NOTE_COLOR)
    fig.savefig(out / "policy_comparison.png", dpi=FIGURE_DPI, facecolor=SURFACE_COLOR)
    plt.close(fig)


# writes summary.csv (one row per policy) and comparison.csv (one row per run)
def write_tables(table, rows, out):
    header = ["policy", "seeds"]
    for metric, title, fmt, limit in METRICS:
        header.append(f"{metric}_mean")
        header.append(f"{metric}_std")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in table:
            seeds = 0
            line = []
            for metric, title, fmt, limit in METRICS:
                mean, std, count = table[key][metric]
                line.append(mean)
                line.append(std)
                if count > seeds:
                    seeds = count
            writer.writerow([key, seeds] + line)

    fields = ["episodes"]
    for metric, title, fmt, limit in METRICS:
        fields.append(metric)
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + fields)
        writer.writeheader()
        for name, report in rows:
            line = {"run": name}
            for field in fields:
                line[field] = report["summary"].get(field)
            writer.writerow(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs", nargs="+", help="Experiment directories containing report.json"
    )
    parser.add_argument("--output", default="docs/figures")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    training, evaluation, rows = load_runs(args.runs)
    if len(rows) == 0:
        raise SystemExit("No report.json found in the given run directories")

    extras = {}
    if len(training) > 0:
        plot_learning_curves(training, out, extras)
    table = summarize(evaluation)
    if len(table) > 0:
        plot_policy_comparison(table, out, extras)
    write_tables(table, rows, out)
    print(f"Wrote figures and tables for {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
