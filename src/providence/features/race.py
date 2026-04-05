"""Race context features."""

from __future__ import annotations

import polars as pl


def add_race_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add race-context features computed within each race.

    Runs AFTER rider features so ``win_rate_10`` etc. are available.
    """
    if df.is_empty():
        return df

    handicap_rank = pl.col("handicap_meters").rank("dense").over("race_id")

    handicap_col = pl.col("handicap_meters")

    df = df.with_columns(
        handicap_rank.alias("handicap_rank"),
        pl.len().over("race_id").alias("field_size"),
        pl.col("trial_time").mean().over("race_id").alias("field_avg_trial"),
        pl.col("trial_time").std().over("race_id").alias("field_trial_std"),
        (handicap_col.max().over("race_id") - handicap_col.min().over("race_id")).alias("handicap_range_in_field"),
        pl.when(handicap_col == 0).then(0)
        .when(handicap_col == 10).then(1)
        .when(handicap_col <= 30).then(2)
        .when(handicap_col <= 50).then(3)
        .otherwise(4)
        .alias("handicap_group"),
    ).with_columns(
        (pl.col("handicap_rank") - pl.col("trial_time_rank")).alias("handicap_vs_trial"),
    )

    if "track_condition" in df.columns:
        tc = pl.col("track_condition")
        is_wet = tc.is_in(["湿", "重"]) if df["track_condition"].dtype == pl.Utf8 else tc.is_in([1, 2])
        df = df.with_columns(is_wet.cast(pl.Int8).alias("is_wet_condition"))

    if "predicted_race_time_rank" in df.columns:
        df = df.with_columns(
            (pl.col("predicted_race_time_rank") - pl.col("handicap_rank")).alias("predicted_vs_handicap"),
        )

    if "win_rate_10" in df.columns:
        df = _add_field_strength_features(df)

    return df


def _add_field_strength_features(df: pl.DataFrame) -> pl.DataFrame:
    """Opponent strength features derived from per-rider win/finish stats."""
    df = df.with_columns(
        pl.col("win_rate_10").mean().over("race_id").alias("field_avg_win_rate"),
        pl.col("win_rate_10").max().over("race_id").alias("field_max_win_rate"),
        pl.col("avg_finish_10").mean().over("race_id").alias("field_avg_finish"),
        pl.col("win_rate_10").rank("dense", descending=True).over("race_id").alias("form_rank_in_field"),
    )

    df = df.with_columns(
        (pl.col("win_rate_10") / pl.col("field_avg_win_rate"))
        .clip(0.0, 10.0)
        .alias("relative_win_strength"),
        (pl.col("field_max_win_rate") - pl.col("win_rate_10")).alias("gap_to_top_rival"),
    )

    return df
