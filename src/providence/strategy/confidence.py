"""Confidence scoring for one race prediction bundle."""

from __future__ import annotations

import math

from providence.strategy.types import RacePredictionBundle


def race_confidence(bundle: RacePredictionBundle) -> float:
    """Return a simple [0, 1] confidence score.

    The first version intentionally uses only stable signals already exposed by
    the feature pipeline: score dispersion and rider history depth.
    """
    if not bundle.scores:
        return 0.0

    max_score = max(bundle.scores)
    min_score = min(bundle.scores)
    score_spread = max_score - min_score
    spread_component = 1.0 - math.exp(-max(score_spread, 0.0))

    if bundle.features_total_races:
        coverage = sum(1 for value in bundle.features_total_races if value >= 5) / len(bundle.features_total_races)
    else:
        coverage = 0.0

    return float(max(0.0, min(1.0, 0.6 * spread_component + 0.4 * coverage)))
