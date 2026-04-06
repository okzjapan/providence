"""Race-level field strength features (within-race relative evaluation)."""

from __future__ import annotations

import polars as pl


def add_field_strength_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add features comparing each horse to the field using IDM."""
    if df.is_empty() or "idm" not in df.columns:
        return df

    df = df.with_columns([
        pl.col("idm").mean().over("race_id").alias("field_avg_idm"),
        pl.col("idm").max().over("race_id").alias("field_max_idm"),
        pl.col("idm").rank("dense", descending=True).over("race_id").alias("idm_rank_in_field"),
    ])

    df = df.with_columns([
        (pl.col("idm") / pl.col("field_avg_idm"))
        .clip(0.0, 10.0)
        .alias("relative_idm"),
        (pl.col("field_max_idm") - pl.col("idm")).alias("gap_to_top"),
    ])

    return df
