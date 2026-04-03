"""Approximate PSI calculation using stored quantile boundaries."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from providence.model.store import ModelStore


def load_feature_stats(store: ModelStore, version: str) -> dict[str, dict]:
    version_dir = store.version_dir(version)
    path = version_dir / "feature_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"feature stats not found for {version}")
    import json

    return json.loads(path.read_text())


def approximate_psi_for_frame(
    feature_df: pl.DataFrame,
    baseline_stats: dict[str, dict],
    *,
    limit_features: int = 10,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for feature_name, stats in baseline_stats.items():
        if len(scores) >= limit_features:
            break
        if feature_name not in feature_df.columns:
            continue
        quantiles = [stats.get("q10"), stats.get("q50"), stats.get("q90")]
        if any(value is None for value in quantiles):
            continue
        series = feature_df[feature_name]
        if not series.dtype.is_numeric():
            continue
        values = series.drop_nulls()
        if values.is_empty():
            continue
        bins = [-math.inf, float(quantiles[0]), float(quantiles[1]), float(quantiles[2]), math.inf]
        counts = [0, 0, 0, 0]
        for value in values.to_list():
            numeric = float(value)
            for idx in range(4):
                if bins[idx] < numeric <= bins[idx + 1]:
                    counts[idx] += 1
                    break
        total = sum(counts)
        if total == 0:
            continue
        actual = [count / total for count in counts]
        expected = [0.25, 0.25, 0.25, 0.25]
        psi = 0.0
        for act, exp in zip(actual, expected, strict=True):
            act = max(act, 1e-6)
            exp = max(exp, 1e-6)
            psi += (act - exp) * math.log(act / exp)
        scores[feature_name] = float(psi)
    return scores
