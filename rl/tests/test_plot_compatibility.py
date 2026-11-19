"""Exercise the learning-curve legend on the container's Matplotlib version."""
import pytest

pytest.importorskip("matplotlib")
from analysis.plot_results import plot_learning_curves


def test_learning_curve_export_with_legend(tmp_path):
    report = {"episodes": [{"episode_return": 0.0, "captured": False}]}
    plot_learning_curves({"full": [report]}, tmp_path, {})
    assert (tmp_path/"learning_curves.png").stat().st_size > 1000
