"""Plot learning curves and baseline comparisons from experiments/*/episodes.csv.

Figures (saved to docs/figures/):
  - capture rate & return vs. episode (mean ± std over seeds)
  - Q-learning vs random vs rule-based: capture rate, time-to-capture, collisions
  - reward ablations: with/without exploration bonus, with/without rediscovery bonus
"""
