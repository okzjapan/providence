"""Running style and pace features."""

from __future__ import annotations

import polars as pl


def add_pace_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add position score, running style, and pace dynamics features."""
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

    import numpy as np

    race_dates = group["race_date"].to_list()
    corner1 = group["corner_1_pos"].to_list() if "corner_1_pos" in group.columns else [None] * len(race_dates)
    corner4 = group["corner_4_pos"].to_list()
    num_runners_list = group["field_size"].to_list()
    last_3f_list = group["last_3f_time"].to_list()

    avg_pos_score: list[float | None] = []
    primary_style: list[int | None] = []
    avg_position_improvement: list[float | None] = []
    best_last_3f_ratio: list[float | None] = []

    pos_history: list[tuple[date, float]] = []
    improvement_history: list[tuple[date, float]] = []
    last_3f_history: list[tuple[date, float]] = []

    for idx, current_date in enumerate(race_dates):
        prior_scores = [s for d, s in pos_history if d < current_date]
        last_5 = prior_scores[-5:]

        if last_5:
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

        prior_improvements = [imp for d, imp in improvement_history if d < current_date]
        last_5_imp = prior_improvements[-5:]
        avg_position_improvement.append(float(np.mean(last_5_imp)) if last_5_imp else None)

        prior_l3 = [l3 for d, l3 in last_3f_history if d < current_date]
        if len(prior_l3) >= 3:
            best = min(prior_l3)
            avg_l3 = np.mean(prior_l3[-5:])
            best_last_3f_ratio.append(float(best / avg_l3) if avg_l3 > 0 else None)
        else:
            best_last_3f_ratio.append(None)

        c1 = corner1[idx]
        c4 = corner4[idx]
        nr = num_runners_list[idx]
        l3 = last_3f_list[idx]

        if c4 is not None and nr is not None and nr > 1:
            score = (c4 - 1) / (nr - 1)
            pos_history.append((current_date, score))

        if c1 is not None and c4 is not None and nr is not None and nr > 1:
            norm_c1 = (c1 - 1) / (nr - 1)
            norm_c4 = (c4 - 1) / (nr - 1)
            improvement = norm_c1 - norm_c4
            improvement_history.append((current_date, improvement))

        if l3 is not None:
            last_3f_history.append((current_date, l3))

    return group.with_columns([
        pl.Series("avg_position_score", avg_pos_score, dtype=pl.Float64),
        pl.Series("primary_running_style", primary_style, dtype=pl.Int64),
        pl.Series("avg_position_improvement", avg_position_improvement, dtype=pl.Float64),
        pl.Series("best_last_3f_ratio", best_last_3f_ratio, dtype=pl.Float64),
    ])
