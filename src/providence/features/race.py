"""Race context features."""

from __future__ import annotations

import polars as pl


def add_race_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add race-context features computed within each race."""
    if df.is_empty():
        return df

    handicap_rank = pl.col("handicap_meters").rank("dense").over("race_id")

    return df.with_columns(
        handicap_rank.alias("handicap_rank"),
        pl.len().over("race_id").alias("field_size"),
        pl.col("trial_time").mean().over("race_id").alias("field_avg_trial"),
        pl.col("trial_time").std().over("race_id").alias("field_trial_std"),
    ).with_columns((pl.col("handicap_rank") - pl.col("trial_time_rank")).alias("handicap_vs_trial"))
