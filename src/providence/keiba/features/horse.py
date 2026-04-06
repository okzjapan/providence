"""Horse attribute and aptitude features."""

from __future__ import annotations

import polars as pl


def add_horse_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add horse-level aptitude and condition-change features."""
    if df.is_empty():
        return df

    df = (
        df.sort(["blood_registration_number", "race_date", "race_number", "post_position"])
        .group_by("blood_registration_number", maintain_order=True)
        .map_groups(_add_horse_group_features)
    )
    return df


def _add_horse_group_features(group: pl.DataFrame) -> pl.DataFrame:
    from datetime import date

    race_dates = group["race_date"].to_list()
    surfaces = group["surface_code"].to_list()
    distances = group["distance"].to_list()
    racecourses = group["racecourse_id"].to_list()
    finish_positions = group["finish_position"].to_list()

    is_first_surface: list[bool | None] = []
    distance_change: list[int | None] = []
    surface_win_rate: list[float | None] = []
    distance_cat_win_rate: list[float | None] = []
    course_win_rate: list[float | None] = []

    history: list[tuple[date, int | None, int | None, int | None, int | None]] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, s, dist, rc, fp) for d, s, dist, rc, fp in history if d < current_date]
        current_surface = surfaces[idx]
        current_dist = distances[idx]
        current_rc = racecourses[idx]

        prior_surfaces = {s for _, s, _, _, _ in prior}
        is_first_surface.append(current_surface not in prior_surfaces if prior else None)

        if prior:
            distance_change.append(current_dist - prior[-1][2] if current_dist and prior[-1][2] else None)
        else:
            distance_change.append(None)

        same_surface = [(fp,) for _, s, _, _, fp in prior if s == current_surface and fp is not None and fp >= 1]
        surface_win_rate.append(
            sum(1 for (fp,) in same_surface if fp == 1) / len(same_surface)
            if len(same_surface) >= 3 else None
        )

        dist_cat = _distance_category(current_dist) if current_dist else None
        same_dist_cat = [
            (fp,) for _, _, d, _, fp in prior
            if _distance_category(d) == dist_cat and fp is not None and fp >= 1
        ] if dist_cat else []
        distance_cat_win_rate.append(
            sum(1 for (fp,) in same_dist_cat if fp == 1) / len(same_dist_cat)
            if len(same_dist_cat) >= 3 else None
        )

        same_course = [(fp,) for _, _, _, rc, fp in prior if rc == current_rc and fp is not None and fp >= 1]
        course_win_rate.append(
            sum(1 for (fp,) in same_course if fp == 1) / len(same_course)
            if len(same_course) >= 3 else None
        )

        fp = finish_positions[idx]
        history.append((current_date, current_surface, current_dist, current_rc, fp))

    return group.with_columns([
        pl.Series("is_first_surface", is_first_surface, dtype=pl.Boolean),
        pl.Series("distance_change", distance_change, dtype=pl.Int64),
        pl.Series("this_surface_win_rate", surface_win_rate, dtype=pl.Float64),
        pl.Series("this_distance_cat_win_rate", distance_cat_win_rate, dtype=pl.Float64),
        pl.Series("this_course_win_rate", course_win_rate, dtype=pl.Float64),
    ])


def _distance_category(dist: int | None) -> str | None:
    if dist is None:
        return None
    if dist <= 1400:
        return "sprint"
    if dist <= 1600:
        return "mile"
    if dist <= 2200:
        return "middle"
    return "long"
