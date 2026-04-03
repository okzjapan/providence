"""Expected value utilities."""

from __future__ import annotations


def compute_expected_value(probability: float, odds_value: float) -> float:
    return float(probability * odds_value - 1.0)
