"""Tabular Q-learning agent (ROS-free).

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Epsilon-greedy over 4 primitive actions, random tie-breaking, no bootstrap on
terminal transitions, linear epsilon decay over episodes.
"""

import os
import hashlib
import io
import json
import tempfile
import warnings
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
    def __init__(self, n_states, alpha, gamma, rng, n_actions=len(ACTIONS), metadata=None):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng
        self.updates = 0
        self.metadata = metadata

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
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, self.q)
        contents = buffer.getvalue()
        metadata = dict(format_version=1, specification=self.metadata,
                        sha256=hashlib.sha256(contents).hexdigest())
        atomic_write(path, contents)
        atomic_write(path.with_suffix(".meta.json"), json.dumps(metadata, indent=2).encode())

    def load(self, path):
        path = Path(path)
        contents = path.read_bytes()
        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint metadata version")
            if metadata["sha256"] != hashlib.sha256(contents).hexdigest():
                raise ValueError("Checkpoint and metadata do not match. Restore a complete saved pair.")
            if self.metadata is not None and metadata.get("specification") != self.metadata:
                raise ValueError("Checkpoint configuration does not match the requested policy configuration")
        elif self.metadata is not None:
            warnings.warn("Legacy Q-table has no configuration metadata. Verify its training configuration.",
                          RuntimeWarning)
        q = np.load(io.BytesIO(contents), allow_pickle=False)
        if q.shape != self.q.shape or not np.isfinite(q).all():
            raise ValueError("Q-table must have the expected shape and finite numeric entries")
        self.q = q.astype(float, copy=False)


def atomic_write(path, contents):
    """Use unique temporary files; a mismatched checkpoint pair is rejected on load."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(contents)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_spec(cfg, kind, explore_cfg=None):
    """Describe policy inputs and actions independently of the training budget."""
    specification = dict(state_encoding_version=1, kind=kind,
        obstacle_bins_m=cfg["obstacle_bins_m"], actions=cfg["actions"],
        action_duration_s=cfg["action_duration_s"])
    if kind == "explore":
        specification.update(stall_bins_steps=explore_cfg.get("stall_bins_steps", [10, 30]),
                             odometry_cell_m=explore_cfg.get("odometry_cell_m", 1.0))
    else:
        specification.update(npc_distance_bins_m=cfg["npc_distance_bins_m"],
            npc_front_half_deg=cfg.get("npc_front_half_deg", 15), detector=cfg["detector"])
    return specification


# linear decay from start to end over decay_episodes, then flat
def epsilon_schedule(episode, start, end, decay_episodes):
    if decay_episodes <= 0:
        return end
    if episode >= decay_episodes:
        return end
    drop = (end - start) * episode / decay_episodes
    return start + drop
