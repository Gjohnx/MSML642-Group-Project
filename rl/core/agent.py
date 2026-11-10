"""Tabular Q-learning agent (ROS-free).

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""

import numpy as np

FORWARD, TURN_LEFT, TURN_RIGHT, STOP = range(4)
ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT, STOP)


class QLearningAgent:
    def __init__(self, n_states: int, alpha: float, gamma: float, rng: np.random.Generator):
        self.q = np.zeros((n_states, len(ACTIONS)))
        self.alpha, self.gamma, self.rng = alpha, gamma, rng

    def act(self, s: int, epsilon: float) -> int:
        raise NotImplementedError

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        np.save(path, self.q)

    def load(self, path: str) -> None:
        self.q = np.load(path)


def epsilon_schedule(episode: int, start: float, end: float, decay_episodes: int) -> float:
    raise NotImplementedError
