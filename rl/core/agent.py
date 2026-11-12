"""Tabular Q-learning agent (ROS-free).

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Epsilon-greedy over 4 primitive actions, random tie-breaking, no bootstrap on
terminal transitions, linear epsilon decay over episodes.
"""

import os
from pathlib import Path

import numpy as np

FORWARD, TURN_LEFT, TURN_RIGHT, STOP = range(4)
ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
ACTION_NAMES = ("forward", "turn_left", "turn_right", "stop")


class QLearningAgent:
    def __init__(self, n_states: int, alpha: float, gamma: float,
                 rng: np.random.Generator, n_actions: int = len(ACTIONS)):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng
        self.updates = 0

    def greedy(self, s: int) -> int:
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def act(self, s: int, epsilon: float) -> tuple[int, bool]:
        """Return (action, explored)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1])), True
        return self.greedy(s), False

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """Apply one Q-learning update and return the TD error."""
        target = r if done else r + self.gamma * float(self.q[s_next].max())
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        self.updates += 1
        return float(td_error)

    def save(self, path) -> None:
        """Write to a temp file, then atomically replace."""
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, self.q)
        os.replace(tmp, path)

    def load(self, path) -> None:
        self.q = np.load(path)


def epsilon_schedule(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Linear decay from start to end over decay_episodes, then flat."""
    if decay_episodes <= 0 or episode >= decay_episodes:
        return end
    return start + (end - start) * episode / decay_episodes
