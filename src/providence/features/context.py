"""Contextual and derived features.

Adds race-time Z-scores from past races, rest-interval segments,
start-timing field-relative position, month/season, and interactions.
"""

from __future__ import annotations

import polars as pl


def add_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add contextual features. Call after prev_race and rider features."""
    if df.is_empty():
        return df

    out = df

    if "race_date" in out.columns:
        out = out.with_columns(pl.col("race_date").dt.month().alias("race_month"))

    if "race_time_prev1" in out.columns:
        out = _add_prev_race_time_zscores(out)

    if "days_since_last_race" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("days_since_last_race") <= 3).then(0)
            .when(pl.col("days_since_last_race") <= 7).then(1)
            .when(pl.col("days_since_last_race") <= 14).then(2)
            .when(pl.col("days_since_last_race") <= 30).then(3)
            .otherwise(4)
            .alias("rest_interval_group")
        )

    if "avg_start_timing_10" in out.columns:
        out = out.with_columns(
            (
                pl.col("avg_start_timing_10")
                - pl.col("avg_start_timing_10").mean().over("race_id")
            ).alias("st_vs_field_avg"),
            pl.col("avg_start_timing_10")
            .rank("dense")
            .over("race_id")
            .alias("st_rank_in_field"),
        )

    if "temperature" in out.columns and "track_condition" in out.columns:
        out = out.with_columns(
            (pl.col("temperature") * pl.col("track_condition")).alias("temp_x_condition"),
        )

    if "humidity" in out.columns and "track_condition" in out.columns:
        out = out.with_columns(
            (pl.col("humidity") * pl.col("track_condition")).alias("humidity_x_condition"),
        )

    if "is_home_track" in out.columns and "rider_track_win_rate" in out.columns:
        out = out.with_columns(
            (pl.col("is_home_track").cast(pl.Float64) * pl.col("rider_track_win_rate")).alias("home_x_track_wr"),
        )

    if "predicted_race_time" in out.columns and "handicap_meters" in out.columns:
        out = out.with_columns(
            (pl.col("handicap_meters").cast(pl.Float64) * 0.05).alias("handicap_seconds"),
        )
        out = out.with_columns(
            (pl.col("predicted_race_time") + pl.col("handicap_seconds")).alias("effective_predicted_time"),
        )
        out = out.with_columns(
            pl.col("effective_predicted_time").rank("dense").over("race_id").alias("effective_time_rank"),
            (pl.col("effective_predicted_time") - pl.col("effective_predicted_time").min().over("race_id")).alias("effective_time_gap_to_best"),
            (pl.col("effective_predicted_time").max().over("race_id") - pl.col("effective_predicted_time").min().over("race_id")).alias("field_effective_time_spread"),
        )

    if "handicap_prev1" in out.columns and "handicap_meters" in out.columns:
        out = out.with_columns(
            (pl.col("handicap_meters") - pl.col("handicap_prev1")).alias("handicap_change_from_prev1"),
        )

    if "win_rate_10" in out.columns and "total_races" in out.columns:
        out = out.with_columns(
            pl.col("win_rate_10").std().over("race_id").alias("field_win_rate_std"),
            pl.col("total_races").mean().over("race_id").alias("field_avg_total_races"),
            (pl.col("total_races").cast(pl.Float64) / pl.col("total_races").mean().over("race_id")).alias("relative_experience"),
        )

    if "momentum" in out.columns:
        out = out.with_columns(
            (pl.col("momentum") - pl.col("momentum").mean().over("race_id")).alias("relative_momentum"),
        )

    if "handicap_meters" in out.columns and "trial_time" in out.columns:
        out = out.with_columns(
            (pl.col("handicap_meters").cast(pl.Float64) * pl.col("trial_time")).alias("handicap_x_trial"),
        )
    if "handicap_meters" in out.columns and "win_rate_10" in out.columns:
        out = out.with_columns(
            (pl.col("handicap_meters").cast(pl.Float64) * pl.col("win_rate_10")).alias("handicap_x_wr"),
        )
    if "trial_time" in out.columns and "avg_start_timing_10" in out.columns:
        out = out.with_columns(
            (pl.col("trial_time") * pl.col("avg_start_timing_10")).alias("trial_x_st"),
        )
    if "post_position" in out.columns and "handicap_meters" in out.columns:
        out = out.with_columns(
            (pl.col("post_position").cast(pl.Float64) * pl.col("handicap_meters").cast(pl.Float64)).alias("post_x_handicap"),
        )

    if "predicted_race_time" in out.columns:
        out = out.with_columns(
            (pl.col("predicted_race_time").max().over("race_id") - pl.col("predicted_race_time").min().over("race_id")).alias("field_predicted_time_spread"),
        )

    return out


def _add_prev_race_time_zscores(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Z-score of past race times relative to their respective field averages.

    Since we don't have the original field stats for past races, we use the
    runner's own time history to approximate a Z-score: (time - mean) / std.
    """
    time_cols = [f"race_time_prev{i}" for i in range(1, 6) if f"race_time_prev{i}" in df.columns]
    if not time_cols:
        return df

    out = df
    for col in time_cols:
        suffix = col.replace("race_time_", "")
        out = out.with_columns(
            pl.when(pl.col("ewm_race_time").is_not_null() & pl.col("prev_time_std").is_not_null() & (pl.col("prev_time_std") > 0))
            .then((pl.col(col) - pl.col("ewm_race_time")) / pl.col("prev_time_std"))
            .otherwise(None)
            .alias(f"time_zscore_{suffix}")
        )

    return out
