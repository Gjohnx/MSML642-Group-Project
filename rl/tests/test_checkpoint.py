"""Check compatibility validation and support for historical Q-tables."""
import numpy as np
import pytest
from rl.core.agent import QLearningAgent


def agent(metadata):
    return QLearningAgent(3, .1, .95, np.random.default_rng(0), metadata=metadata)


def test_checkpoint_metadata_roundtrip_and_mismatch(tmp_path):
    path = tmp_path/"q.npy"
    original = agent({"state_encoding_version": 1, "speed": .35})
    original.q[0, 0] = 2
    original.save(path)
    restored = agent(original.metadata)
    restored.load(path)
    assert np.array_equal(original.q, restored.q)
    with pytest.raises(ValueError, match="configuration"):
        agent({"state_encoding_version": 1, "speed": .25}).load(path)
    np.save(path, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="complete saved pair"):
        restored.load(path)


def test_legacy_checkpoint_and_invalid_values(tmp_path):
    path = tmp_path/"legacy.npy"
    np.save(path, np.zeros((3, 4)))
    with pytest.warns(RuntimeWarning, match="Legacy"):
        agent({"version": 1}).load(path)
    np.save(path, np.full((3, 4), np.nan))
    with pytest.raises(ValueError, match="finite"):
        agent(None).load(path)
