"""Running style and pace features."""

from __future__ import annotations

import polars as pl


def add_pace_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add position score and running style features."""
    if df.is_empty():
        return df

    df = (
        df.sort(["blood_registration_number", "race_date", "race_number", "post_position"])
        .group_by("blood_registration_number", maintain_order=True)
        .map_groups(_add_pace_group_features)
    )
    return df


def _add_pace_group_features(group: pl.DataFrame) -> pl.DataFrame:
    from datetime import date

    race_dates = group["race_date"].to_list()
    corner4 = group["corner_4_pos"].to_list()
    num_runners_list = group["field_size"].to_list()

    avg_pos_score: list[float | None] = []
    primary_style: list[int | None] = []

    history: list[tuple[date, float]] = []

    for idx, current_date in enumerate(race_dates):
        prior_scores = [s for d, s in history if d < current_date]
        last_5 = prior_scores[-5:]

        if last_5:
            import numpy as np
            avg_pos_score.append(float(np.mean(last_5)))
            avg = np.mean(last_5)
            if avg < 0.15:
                primary_style.append(1)
            elif avg < 0.35:
                primary_style.append(2)
            elif avg < 0.65:
                primary_style.append(3)
            else:
                primary_style.append(4)
        else:
            avg_pos_score.append(None)
            primary_style.append(None)

        c4 = corner4[idx]
        nr = num_runners_list[idx]
        if c4 is not None and nr is not None and nr > 1:
            score = (c4 - 1) / (nr - 1)
            history.append((current_date, score))

    return group.with_columns([
        pl.Series("avg_position_score", avg_pos_score, dtype=pl.Float64),
        pl.Series("primary_running_style", primary_style, dtype=pl.Int64),
    ])
