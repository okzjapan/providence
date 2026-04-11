"""Ensemble scoring: combine multiple LightGBM models into unified race scores."""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {
    "lambdarank": 0.30,
    "xendcg": 0.15,
    "binary_top2": 0.20,
    "focal_value": 0.15,
    "binary_win": 0.10,
    "huber": 0.10,
}

MODEL_KEYS = ("lambdarank", "xendcg", "binary_top2", "focal_value", "binary_win", "huber")


def combine_race_scores(
    raw_outputs: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine per-model raw outputs for one race into unified log-scores.

    Each model's output is converted to a probability distribution over runners,
    then combined via weighted geometric mean.  The result is returned as
    log-probabilities suitable for Plackett-Luce (temperature=1).

    Parameters
    ----------
    raw_outputs:
        ``{model_key: 1-D array of raw predictions for N runners}``.
        LambdaRank/Huber produce arbitrary scores; Binary models produce
        probabilities in [0, 1].
    weights:
        ``{model_key: float}``.  Missing keys default to 0.
    """
    weights = weights or DEFAULT_WEIGHTS
    n = _validate_and_get_n(raw_outputs)

    log_combined = np.zeros(n, dtype=float)
    total_weight = 0.0

    for key in MODEL_KEYS:
        raw = raw_outputs.get(key)
        w = weights.get(key, 0.0)
        if raw is None or w <= 0:
            continue
        probs = _to_race_probabilities(raw, key)
        log_combined += w * np.log(np.maximum(probs, 1e-12))
        total_weight += w

    if total_weight <= 0:
        return np.zeros(n, dtype=float)

    log_combined /= total_weight
    return log_combined


def combined_scores_to_temperature(
    log_scores: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert combined log-scores to (scores_array, temperature=1.0) for PL."""
    return log_scores, 1.0


def _to_race_probabilities(raw: np.ndarray, model_key: str) -> np.ndarray:
    """Normalize raw model output to a probability distribution over runners."""
    arr = np.asarray(raw, dtype=float)
    if model_key in ("binary_top2", "binary_win"):
        clipped = np.clip(arr, 1e-8, 1.0 - 1e-8)
        total = clipped.sum()
        return clipped / total if total > 0 else np.ones_like(clipped) / len(clipped)
    if model_key == "focal_value":
        from scipy.special import expit
        probs = expit(arr)
        clipped = np.clip(probs, 1e-8, 1.0 - 1e-8)
        total = clipped.sum()
        return clipped / total if total > 0 else np.ones_like(clipped) / len(clipped)
    return _softmax(arr)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exps = np.exp(shifted)
    total = exps.sum()
    return exps / total if total > 0 else np.ones_like(exps) / len(exps)


def _validate_and_get_n(raw_outputs: dict[str, np.ndarray]) -> int:
    sizes = [len(v) for v in raw_outputs.values() if v is not None]
    if not sizes:
        raise ValueError("No model outputs provided")
    n = sizes[0]
    if any(s != n for s in sizes):
        raise ValueError(f"Inconsistent output sizes: {sizes}")
    return n
