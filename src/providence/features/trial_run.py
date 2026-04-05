"""Trial run feature engineering."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

_MIN_DEVIATION_SAMPLES = 5


def add_trial_run_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add trial-time based features."""
    if df.is_empty():
        return df

    out = df.with_columns(
        pl.col("trial_time").rank("dense").over("race_id").alias("trial_time_rank"),
        pl.col("trial_time").min().over("race_id").alias("_best_trial_time"),
        pl.col("trial_time").mean().over("race_id").alias("field_avg_trial"),
        pl.col("trial_time").std().over("race_id").alias("field_trial_std"),
    ).with_columns(
        (pl.col("trial_time") - pl.col("_best_trial_time")).alias("trial_time_diff_from_best")
    ).drop("_best_trial_time")

    per_rider = (
        out.sort(["rider_id", "race_date", "race_number", "post_position"])
        .group_by("rider_id", maintain_order=True)
        .map_groups(_add_rider_trial_history_features)
    )

    per_rider = per_rider.with_columns(
        (pl.col("trial_time") + pl.col("computed_trial_deviation")).alias("predicted_race_time"),
    )
    per_rider = per_rider.with_columns(
        pl.col("predicted_race_time").rank("dense").over("race_id").alias("predicted_race_time_rank"),
    )

    if "field_trial_std" in per_rider.columns:
        std_col = pl.col("field_trial_std")
        per_rider = per_rider.with_columns(
            pl.when(std_col > 0)
            .then((pl.col("trial_time") - pl.col("field_avg_trial")) / std_col)
            .otherwise(0.0)
            .alias("trial_time_zscore"),
        )

    return per_rider


def _add_rider_trial_history_features(group: pl.DataFrame) -> pl.DataFrame:
    trial_times = group["trial_time"].to_list()
    race_dates: Sequence[date] = group["race_date"].to_list()
    race_times = group["race_time"].to_list() if "race_time" in group.columns else [None] * len(trial_times)
    track_conditions = (
        group["track_condition"].to_list() if "track_condition" in group.columns else [None] * len(trial_times)
    )

    avg_30d: list[float | None] = []
    avg_90d: list[float | None] = []
    trend_5: list[float | None] = []
    vs_avg_90d: list[float | None] = []
    computed_td: list[float | None] = []
    td_trend: list[float | None] = []

    trial_history: list[tuple[date, float]] = []
    deviation_history: list[float] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, t) for d, t in trial_history if d < current_date]
        valid_30 = [t for d, t in prior if (current_date - d).days <= 30]
        valid_90 = [t for d, t in prior if (current_date - d).days <= 90]
        last_5_trial = [t for _, t in prior[-5:]]

        avg30 = _mean_or_none(valid_30)
        avg90 = _mean_or_none(valid_90)
        trend = _trend_or_none(last_5_trial)
        current_trial = trial_times[idx]

        avg_30d.append(avg30)
        avg_90d.append(avg90)
        trend_5.append(trend)
        vs_avg_90d.append((current_trial - avg90) if current_trial is not None and avg90 is not None else None)

        prior_dev_count = len(deviation_history)
        last_10_dev = deviation_history[-10:] if prior_dev_count > 0 else []
        last_5_dev = deviation_history[-5:] if prior_dev_count > 0 else []

        computed_td.append(_mean_or_none(last_10_dev) if len(last_10_dev) >= _MIN_DEVIATION_SAMPLES else None)
        td_trend.append(_trend_or_none(last_5_dev))

        if current_trial is not None:
            trial_history.append((current_date, current_trial))

        current_race_time = race_times[idx]
        current_tc = track_conditions[idx]
        if current_trial is not None and current_race_time is not None and current_tc == "良":
            deviation_history.append(current_race_time - current_trial)

    return group.with_columns(
        pl.Series("avg_trial_time_30d", avg_30d),
        pl.Series("avg_trial_time_90d", avg_90d),
        pl.Series("trial_time_trend", trend_5),
        pl.Series("trial_vs_avg_90d", vs_avg_90d),
        pl.Series("computed_trial_deviation", computed_td),
        pl.Series("trial_deviation_trend", td_trend),
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _trend_or_none(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)
