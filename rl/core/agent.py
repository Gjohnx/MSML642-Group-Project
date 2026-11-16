"""Tabular Q-learning agent (ROS-free).

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Epsilon-greedy over 4 primitive actions, random tie-breaking, no bootstrap on
terminal transitions, linear epsilon decay over episodes.
"""

import os
from pathlib import Path

import numpy as np

# robot actions
FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
STOP = 3
ACTIONS = (FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
ACTION_NAMES = ("forward", "turn_left", "turn_right", "stop")


# tabular Q-learning over the four robot actions
class QLearningAgent:
    def __init__(self, n_states, alpha, gamma, rng, n_actions=len(ACTIONS)):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng
        self.updates = 0

    # picks the best action for a state, ties are broken at random
    def greedy(self, s):
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        action = self.rng.choice(best)
        return int(action)

    # returns the chosen action and whether it was an exploration step
    def act(self, s, epsilon):
        explored = False
        if self.rng.random() < epsilon:
            n_actions = self.q.shape[1]
            action = int(self.rng.integers(n_actions))
            explored = True
        else:
            action = self.greedy(s)
        return action, explored

    # applies one Q-learning update and returns the TD error
    def update(self, s, a, r, s_next, done):
        target = r
        if not done:
            best_next = float(self.q[s_next].max())
            target = r + self.gamma * best_next
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        self.updates += 1
        return float(td_error)

    # writes to a temp file first, then replaces the old table
    def save(self, path):
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, self.q)
        os.replace(tmp, path)

    def load(self, path):
        self.q = np.load(path)


# linear decay from start to end over decay_episodes, then flat
def epsilon_schedule(episode, start, end, decay_episodes):
    if decay_episodes <= 0:
        return end
    if episode >= decay_episodes:
        return end
    drop = (end - start) * episode / decay_episodes
    return start + drop
