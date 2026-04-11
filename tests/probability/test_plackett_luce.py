"""Tests for Plackett-Luce probability utilities."""

import numpy as np

from providence.probability.calibration import TemperatureScaler
from providence.probability.plackett_luce import (
    compute_all_ticket_probs,
    compute_exacta_probs,
    compute_quinella_probs,
    compute_trifecta_probs,
    compute_trio_probs,
    compute_wide_probs,
    compute_win_probs,
)


def test_win_probs_sum_to_one():
    scores = np.array([1.0, 0.5, -0.2])
    probs = compute_win_probs(scores)
    assert np.isclose(probs.sum(), 1.0)


def test_exacta_probs_sum_to_one():
    scores = np.array([1.0, 0.5, -0.2])
    probs = compute_exacta_probs(scores)
    assert np.isclose(sum(probs.values()), 1.0)


def test_trifecta_probs_sum_to_one():
    scores = np.array([1.0, 0.5, -0.2])
    probs = compute_trifecta_probs(scores)
    assert np.isclose(sum(probs.values()), 1.0)


def test_quinella_and_trio_counts():
    scores = np.array([1.0, 0.5, -0.2, 0.1])
    assert len(compute_quinella_probs(scores)) == 6
    assert len(compute_trio_probs(scores)) == 4
    assert len(compute_wide_probs(scores)) == 6


def test_compute_all_ticket_probs_contains_all_types():
    scores = np.array([1.0, 0.5, 0.2, -0.1])
    probs = compute_all_ticket_probs(scores)
    assert set(probs) == {"win", "place", "exacta", "quinella", "trifecta", "trio", "wide"}


def test_temperature_scaler_identity_transform():
    scaler = TemperatureScaler(temperature=1.0)
    scores = np.array([1.0, 2.0, 3.0])
    transformed = scaler.transform(scores)
    assert np.allclose(transformed, scores)
