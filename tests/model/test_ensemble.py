"""Tests for ensemble scoring logic."""

from __future__ import annotations

import numpy as np
import pytest

from providence.model.ensemble import (
    DEFAULT_WEIGHTS,
    combine_race_scores,
)


def test_combine_produces_valid_log_probabilities():
    raw = {
        "lambdarank": np.array([2.0, 1.0, 0.5]),
        "binary_top2": np.array([0.6, 0.3, 0.1]),
        "binary_win": np.array([0.5, 0.3, 0.2]),
        "huber": np.array([8.0, 4.0, 2.0]),
    }
    log_scores = combine_race_scores(raw, DEFAULT_WEIGHTS)
    assert len(log_scores) == 3
    probs = np.exp(log_scores - log_scores.max())
    probs /= probs.sum()
    assert abs(probs.sum() - 1.0) < 1e-6
    assert probs[0] > probs[1] > probs[2], "Stronger runner should have higher probability"


def test_combine_with_single_model():
    raw = {"lambdarank": np.array([3.0, 1.0, 0.0])}
    weights = {"lambdarank": 1.0}
    log_scores = combine_race_scores(raw, weights)
    probs = np.exp(log_scores - log_scores.max())
    probs /= probs.sum()
    assert probs[0] > probs[1] > probs[2]


def test_combine_raises_on_inconsistent_sizes():
    raw = {
        "lambdarank": np.array([1.0, 2.0]),
        "binary_win": np.array([0.3, 0.4, 0.3]),
    }
    with pytest.raises(ValueError, match="Inconsistent"):
        combine_race_scores(raw)


def test_combine_raises_on_empty():
    with pytest.raises(ValueError, match="No model outputs"):
        combine_race_scores({})


def test_combine_ignores_zero_weight_models():
    raw = {
        "lambdarank": np.array([3.0, 1.0]),
        "binary_top2": np.array([0.1, 0.9]),
    }
    weights = {"lambdarank": 1.0, "binary_top2": 0.0}
    log_scores = combine_race_scores(raw, weights)
    probs = np.exp(log_scores - log_scores.max())
    probs /= probs.sum()
    assert probs[0] > probs[1], "Only lambdarank (favoring runner 0) should contribute"
