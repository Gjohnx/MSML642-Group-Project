"""Evaluate a policy: `python3 -m rl.evaluate --policy {<q_table.npy>|random|rule}`.

Runs greedy episodes over the spawn/route grid and writes a metrics CSV:
capture rate, time to first detection, time to capture, collisions,
rediscovery rate, average episode reward.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
