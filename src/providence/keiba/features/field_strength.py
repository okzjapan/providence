"""Race-level field strength features (within-race relative evaluation)."""

from __future__ import annotations

import polars as pl


def add_field_strength_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add features comparing each horse to the field using multiple indices."""
    if df.is_empty() or "idm" not in df.columns:
        return df

    df = df.with_columns([
        pl.col("idm").mean().over("race_id").alias("field_avg_idm"),
        pl.col("idm").max().over("race_id").alias("field_max_idm"),
        pl.col("idm").std().over("race_id").alias("field_std_idm"),
        pl.col("idm").rank("dense", descending=True).over("race_id").alias("idm_rank_in_field"),
    ])

    df = df.with_columns([
        (pl.col("idm") / pl.col("field_avg_idm"))
        .clip(0.0, 10.0)
        .alias("relative_idm"),
        (pl.col("field_max_idm") - pl.col("idm")).alias("gap_to_top"),
    ])

    if "composite_index" in df.columns:
        df = df.with_columns([
            pl.col("composite_index").rank("dense", descending=True).over("race_id").alias("composite_rank"),
        ])

    if "jockey_index" in df.columns:
        df = df.with_columns([
            pl.col("jockey_index").rank("dense", descending=True).over("race_id").alias("jockey_index_rank"),
        ])

    if "training_index" in df.columns:
        df = df.with_columns([
            pl.col("training_index").rank("dense", descending=True).over("race_id").alias("training_rank"),
        ])

    if "ten_index_pred" in df.columns:
        df = df.with_columns([
            pl.col("ten_index_pred").rank("dense", descending=True).over("race_id").alias("ten_pred_rank"),
        ])

    if "agari_index_pred" in df.columns:
        df = df.with_columns([
            pl.col("agari_index_pred").rank("dense", descending=True).over("race_id").alias("agari_pred_rank"),
        ])

    if "position_index_pred" in df.columns:
        df = df.with_columns([
            pl.col("position_index_pred").rank("dense", descending=True).over("race_id").alias("position_pred_rank"),
        ])

    for z_col, rank_name in [
        ("avg_z_time_5", "z_time_rank"),
        ("avg_z_last3f_5", "z_last3f_rank"),
        ("avg_z_idm_5", "z_idm_rank"),
        ("form_score_z", "form_score_z_rank"),
    ]:
        if z_col in df.columns:
            df = df.with_columns([
                pl.col(z_col).rank("dense", descending=False).over("race_id").alias(rank_name),
            ])

    return df
