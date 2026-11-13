"""Train tabular Q-learning on the Gazebo search-and-tracking task."""
from rl.runner import parser, run


def main():
    run(parser(training=True).parse_args(), training=True)


if __name__ == "__main__":
    main()
