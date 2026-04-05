"""Track and environment features."""

from __future__ import annotations

from datetime import date

import polars as pl


def add_track_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rider-track and rider-wet historical performance features."""
    if df.is_empty():
        return df

    grouped = (
        df.sort(["rider_id", "race_date", "race_number", "post_position"])
        .group_by("rider_id", maintain_order=True)
        .map_groups(_add_rider_track_group_features)
    )

    return grouped


def _add_rider_track_group_features(group: pl.DataFrame) -> pl.DataFrame:
    track_ids = group["track_id"].to_list()
    finish_positions = group["finish_position"].to_list()
    track_conditions = group["track_condition"].to_list()

    rider_track_win_rate: list[float | None] = []
    rider_wet_win_rate: list[float | None] = []
    rider_wet_top3_rate: list[float | None] = []

    race_dates = group["race_date"].to_list()
    history: list[tuple[date, int, str | None, int]] = []

    for idx, track_id in enumerate(track_ids):
        current_date = race_dates[idx]
        prior_history = [(d, t, c, p) for d, t, c, p in history if d < current_date]
        starts = sum(1 for _, t, _, _ in prior_history if t == track_id)
        wins = sum(1 for _, t, _, p in prior_history if t == track_id and p == 1)
        rider_track_win_rate.append((wins / starts) if starts >= 5 else None)
        wet_history = [p for _, _, c, p in prior_history if c in {"湿", "重"}]
        wet_starts = len(wet_history)
        wet_wins = sum(1 for p in wet_history if p == 1)
        rider_wet_win_rate.append((wet_wins / wet_starts) if wet_starts >= 5 else None)
        wet_top3 = sum(1 for p in wet_history if p <= 3)
        rider_wet_top3_rate.append((wet_top3 / wet_starts) if wet_starts >= 5 else None)

        position = finish_positions[idx]
        if position is not None and position >= 1:
            history.append((current_date, track_id, track_conditions[idx], int(position)))

    return group.with_columns(
        pl.Series("rider_track_win_rate", rider_track_win_rate),
        pl.Series("rider_wet_win_rate", rider_wet_win_rate),
        pl.Series("rider_wet_top3_rate", rider_wet_top3_rate),
    )
