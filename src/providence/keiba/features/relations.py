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


def _add_person_stats(prefix: str):
    def _inner(group: pl.DataFrame) -> pl.DataFrame:
        from datetime import date

        race_dates = group["race_date"].to_list()
        finish_positions = group["finish_position"].to_list()
        racecourses = group["racecourse_id"].to_list()

        win_rate: list[float | None] = []
        top3_rate: list[float | None] = []
        course_win_rate: list[float | None] = []

        history: list[tuple[date, int | None, int | None]] = []

        for idx, current_date in enumerate(race_dates):
            prior = [(d, fp, rc) for d, fp, rc in history if d < current_date]
            fps = [fp for _, fp, _ in prior if fp is not None]
            last_30 = fps[-30:]

            win_rate.append(sum(1 for x in last_30 if x == 1) / len(last_30) if last_30 else None)
            top3_rate.append(sum(1 for x in last_30 if x <= 3) / len(last_30) if last_30 else None)

            current_rc = racecourses[idx]
            same_course_fps = [fp for _, fp, rc in prior if rc == current_rc and fp is not None]
            course_win_rate.append(
                sum(1 for x in same_course_fps if x == 1) / len(same_course_fps)
                if len(same_course_fps) >= 10 else None
            )

            fp = finish_positions[idx]
            rc = racecourses[idx]
            if fp is not None and fp >= 1:
                history.append((current_date, int(fp), rc))

        return group.with_columns([
            pl.Series(f"{prefix}_win_rate", win_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_top3_rate", top3_rate, dtype=pl.Float64),
            pl.Series(f"{prefix}_course_win_rate", course_win_rate, dtype=pl.Float64),
        ])

    return _inner


def _add_sire_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add sire-level win rates using sire_code (system-level, not individual stallion)."""
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

    sire_win_rate: list[float | None] = []
    sire_surface_win_rate: list[float | None] = []

    history: list[tuple[date, int | None, int | None]] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, fp, s) for d, fp, s in history if d < current_date]
        fps = [fp for _, fp, _ in prior if fp is not None]

        sire_win_rate.append(
            sum(1 for x in fps if x == 1) / len(fps) if len(fps) >= 20 else None
        )

        current_surface = surfaces[idx]
        same_surface_fps = [fp for _, fp, s in prior if s == current_surface and fp is not None]
        sire_surface_win_rate.append(
            sum(1 for x in same_surface_fps if x == 1) / len(same_surface_fps)
            if len(same_surface_fps) >= 10 else None
        )

        fp = finish_positions[idx]
        s = surfaces[idx]
        if fp is not None and fp >= 1:
            history.append((current_date, int(fp), s))

    return group.with_columns([
        pl.Series("sire_win_rate", sire_win_rate, dtype=pl.Float64),
        pl.Series("sire_surface_win_rate", sire_surface_win_rate, dtype=pl.Float64),
    ])
