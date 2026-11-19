"""Reject overlapping controllers and release the lock after an error."""
import pytest

pytest.importorskip("fcntl")
from rl.ros.control_lock import exclusive_control


def test_controller_lock_rejects_overlap_and_releases(tmp_path):
    path = tmp_path/"controller.lock"
    with exclusive_control("demo", path):
        with pytest.raises(RuntimeError, match="Stop that controller"):
            with exclusive_control("training", path):
                pass
    with pytest.raises(ValueError):
        with exclusive_control("evaluation", path):
            raise ValueError("test")
    with exclusive_control("training", path):
        assert "training" in path.read_text()
