"""Train the Q-learning agent against the running sim.

    docker compose exec sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /project && \
        python3 -m rl.train --config rl/config/qlearning.yaml --rewards rl/config/rewards.yaml --seed 0'

Writes experiments/<date>_<name>_seed<k>/: config copy, q_table.npy,
episodes.csv (captured, steps, collisions, time_to_detect, rediscoveries, return).
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
