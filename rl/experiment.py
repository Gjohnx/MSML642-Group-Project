"""Run equal-budget policies and reward ablations with reproducible seeds."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--output", default="experiments/suite")
    p.add_argument("--ablations", action="store_true")
    args = p.parse_args()
    root = Path(args.output)
    plots = []
    def command(module, *options):
        subprocess.run([sys.executable, "-m", module, "--max-steps", str(args.max_steps),
                        *map(str, options)], check=True)
    for seed in args.seeds:
        for variant, flags in [("full", []), *([("no_exploration", ["--no-exploration-bonus"]),
                    ("no_rediscovery", ["--no-rediscovery-bonus"])] if args.ablations else [])]:
            train = root/f"train_{variant}_seed{seed}"
            command("rl.train", "--seed", seed, "--episodes", args.episodes,
                    "--epsilon-decay-episodes", max(1, int(args.episodes * 0.75)), "--output", train, *flags)
            evaluate = root/f"eval_{variant}_seed{seed}"
            command("rl.evaluate", "--seed", seed, "--episodes", args.eval_episodes,
                    "--policy", train/"q_table.npy", "--output", evaluate, *flags)
            plots.extend([train, evaluate])
        for policy in ("random", "rule"):
            evaluate = root/f"eval_{policy}_seed{seed}"
            command("rl.evaluate", "--seed", seed, "--episodes", args.eval_episodes,
                    "--policy", policy, "--output", evaluate)
            plots.append(evaluate)
    subprocess.run([sys.executable, "-m", "analysis.plot_results", *map(str, plots),
                    "--output", str(root/"figures")], check=True)


if __name__ == "__main__":
    main()
