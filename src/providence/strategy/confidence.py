"""Confidence scoring for one race prediction bundle."""

from __future__ import annotations

import math

from providence.strategy.types import RacePredictionBundle


def race_confidence(bundle: RacePredictionBundle) -> float:
    """Return a [0, 1] confidence score.

    Three components weighted:
    - Score spread: larger dispersion = more decisive prediction
    - Rider history coverage: more experienced riders = more reliable features
    - Trial time coverage: more trial times = core input data available
    """
    if not bundle.scores:
        return 0.0

    max_score = max(bundle.scores)
    min_score = min(bundle.scores)
    score_spread = max_score - min_score
    spread_component = 1.0 - math.exp(-max(score_spread, 0.0))

    n = len(bundle.scores)
    if bundle.features_total_races:
        history_coverage = sum(1 for value in bundle.features_total_races if value >= 5) / len(bundle.features_total_races)
    else:
        history_coverage = 0.0

    trial_coverage = 1.0
    if bundle.features_trial_available is not None and n > 0:
        trial_coverage = sum(bundle.features_trial_available) / n

    return float(max(0.0, min(1.0, 0.5 * spread_component + 0.3 * history_coverage + 0.2 * trial_coverage)))
