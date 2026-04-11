"""Jockey, trainer, and bloodline relation features."""

from __future__ import annotations

import polars as pl


def add_relation_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add jockey/trainer/sire historical performance features."""
    if df.is_empty():
        return df

    df = _add_jockey_features(df)
    df = _add_trainer_features(df)
    df = _add_sire_features(df)
    return df


def _add_jockey_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["jockey_code", "race_date", "race_number", "post_position"])
        .group_by("jockey_code", maintain_order=True)
        .map_groups(_add_person_stats("jockey"))
    )


def _add_trainer_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["trainer_code", "race_date", "race_number", "post_position"])
        .group_by("trainer_code", maintain_order=True)
        .map_groups(_add_person_stats("trainer"))
    )


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


def _add_person_stats(prefix: str):
    def _inner(group: pl.DataFrame) -> pl.DataFrame:
        from datetime import date

        race_dates = group["race_date"].to_list()
        finish_positions = group["finish_position"].to_list()
        racecourses = group["racecourse_id"].to_list()
        distances = group["distance"].to_list()
        surfaces = group["surface_code"].to_list()

        win_rate: list[float | None] = []
        top3_rate: list[float | None] = []
        course_win_rate: list[float | None] = []
        distance_win_rate: list[float | None] = []
        surface_win_rate: list[float | None] = []

        history: list[tuple[date, int | None, int | None, str | None, int | None]] = []

        for idx, current_date in enumerate(race_dates):
            prior = [(d, fp, rc, dc, sc) for d, fp, rc, dc, sc in history if d < current_date]
            fps = [fp for _, fp, _, _, _ in prior if fp is not None]
            last_30 = fps[-30:]

            win_rate.append(sum(1 for x in last_30 if x == 1) / len(last_30) if last_30 else None)
            top3_rate.append(sum(1 for x in last_30 if x <= 3) / len(last_30) if last_30 else None)

            current_rc = racecourses[idx]
            same_course_fps = [fp for _, fp, rc, _, _ in prior if rc == current_rc and fp is not None]
            course_win_rate.append(
                sum(1 for x in same_course_fps if x == 1) / len(same_course_fps)
                if len(same_course_fps) >= 10 else None
            )

            current_dist_cat = _distance_category(distances[idx])
            if current_dist_cat:
                same_dist_fps = [
                    fp for _, fp, _, dc, _ in prior
                    if dc == current_dist_cat and fp is not None
                ]
                distance_win_rate.append(
                    sum(1 for x in same_dist_fps if x == 1) / len(same_dist_fps)
                    if len(same_dist_fps) >= 10 else None
                )
            else:
                distance_win_rate.append(None)

            current_surface = surfaces[idx]
            same_surface_fps = [fp for _, fp, _, _, sc in prior if sc == current_surface and fp is not None]
            surface_win_rate.append(
                sum(1 for x in same_surface_fps if x == 1) / len(same_surface_fps)
                if len(same_surface_fps) >= 10 else None
            )

            fp = finish_positions[idx]
            rc = racecourses[idx]
            dc = _distance_category(distances[idx])
            sc = surfaces[idx]
            if fp is not None and fp >= 1:
                history.append((current_date, int(fp), rc, dc, sc))

        return group.with_columns([
            pl.Series(f"{prefix}_win_rate", win_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_top3_rate", top3_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_course_win_rate", course_win_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_distance_win_rate", distance_win_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_surface_win_rate", surface_win_rate, dtype=pl.Float64),
        ])

    return _inner


def _add_sire_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add sire-level win rates using sire_code."""
    if "sire_code" not in df.columns:
        return df

    return (
        df.sort(["sire_code", "race_date", "race_number", "post_position"])
        .group_by("sire_code", maintain_order=True)
        .map_groups(_add_sire_group_features)
    )


def _add_sire_group_features(group: pl.DataFrame) -> pl.DataFrame:
    from datetime import date

    race_dates = group["race_date"].to_list()
    finish_positions = group["finish_position"].to_list()
    surfaces = group["surface_code"].to_list()
    distances = group["distance"].to_list()

    sire_win_rate: list[float | None] = []
    sire_surface_win_rate: list[float | None] = []
    sire_distance_win_rate: list[float | None] = []

    history: list[tuple[date, int | None, int | None, str | None]] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, fp, s, dc) for d, fp, s, dc in history if d < current_date]
        fps = [fp for _, fp, _, _ in prior if fp is not None]

        sire_win_rate.append(
            sum(1 for x in fps if x == 1) / len(fps) if len(fps) >= 20 else None
        )

        current_surface = surfaces[idx]
        same_surface_fps = [fp for _, fp, s, _ in prior if s == current_surface and fp is not None]
        sire_surface_win_rate.append(
            sum(1 for x in same_surface_fps if x == 1) / len(same_surface_fps)
            if len(same_surface_fps) >= 10 else None
        )

        current_dist_cat = _distance_category(distances[idx])
        if current_dist_cat:
            same_dist_fps = [fp for _, fp, _, dc in prior if dc == current_dist_cat and fp is not None]
            sire_distance_win_rate.append(
                sum(1 for x in same_dist_fps if x == 1) / len(same_dist_fps)
                if len(same_dist_fps) >= 10 else None
            )
        else:
            sire_distance_win_rate.append(None)

        fp = finish_positions[idx]
        s = surfaces[idx]
        dc = _distance_category(distances[idx])
        if fp is not None and fp >= 1:
            history.append((current_date, int(fp), s, dc))

    return group.with_columns([
        pl.Series("sire_win_rate", sire_win_rate, dtype=pl.Float64),
        pl.Series("sire_surface_win_rate", sire_surface_win_rate, dtype=pl.Float64),
        pl.Series("sire_distance_win_rate", sire_distance_win_rate, dtype=pl.Float64),
    ])
