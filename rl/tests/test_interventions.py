"""Keep intervention metrics compatible with historical experiment reports."""
from rl.metrics import EpisodeMetrics
from rl.evaluate import summarize


def test_intervention_counts_and_rates():
    metrics = EpisodeMetrics()
    metrics.observe_step(False, 0, False, False, shielded=True, search_override=True)
    metrics.observe_step(True, 0, True, False, recovered=True)
    row = metrics.to_dict()
    summary = summarize([row])
    assert row["shielded_steps"] == row["search_steps"] == row["recovery_events"] == 1
    assert summary["mean_shielded_rate"] == summary["mean_search_override_rate"] == .5
    assert summary["mean_recovery_events"] == 1
    for key in ("shielded_steps", "search_steps", "recovery_events"):
        row.pop(key)
    assert summarize([row])["mean_shielded_rate"] is None
