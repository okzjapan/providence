"""Horse performance features based on past race history."""

from __future__ import annotations

import polars as pl


def add_performance_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-horse historical performance features (time-series safe)."""
    if df.is_empty():
        return df
    return (
        df.sort(["blood_registration_number", "race_date", "race_number", "post_position"])
        .group_by("blood_registration_number", maintain_order=True)
        .map_groups(_add_horse_performance)
    )


def _add_horse_performance(group: pl.DataFrame) -> pl.DataFrame:
    from collections.abc import Sequence
    from datetime import date

    import numpy as np

    race_dates: Sequence[date] = group["race_date"].to_list()
    finish_positions = group["finish_position"].to_list()
    last_3f_times = group["last_3f_time"].to_list()
    race_time_secs = group["race_time_sec"].to_list() if "race_time_sec" in group.columns else [None] * len(race_dates)

    prev_finish: list[list[float | None]] = [[] for _ in range(5)]
    prev_last_3f: list[list[float | None]] = [[] for _ in range(5)]
    prev_time: list[list[float | None]] = [[] for _ in range(5)]
    win_rate_10: list[float | None] = []
    top3_rate_10: list[float | None] = []
    avg_finish_10: list[float | None] = []
    avg_last_3f_5: list[float | None] = []
    ewm_finish: list[float | None] = []
    days_since: list[int | None] = []
    total_races: list[int] = []

    history: list[tuple[date, int | None, float | None, float | None]] = []

    for idx, current_date in enumerate(race_dates):
        prior = [(d, fp, l3, rt) for d, fp, l3, rt in history if d < current_date]
        fps = [fp for _, fp, _, _ in prior if fp is not None]
        l3s = [l3 for _, _, l3, _ in prior if l3 is not None]
        rts = [rt for _, _, _, rt in prior if rt is not None]

        for n in range(5):
            prev_finish[n].append(fps[-(n + 1)] if len(fps) > n else None)
            prev_last_3f[n].append(l3s[-(n + 1)] if len(l3s) > n else None)
            prev_time[n].append(rts[-(n + 1)] if len(rts) > n else None)

        last_10 = fps[-10:]
        win_rate_10.append(sum(1 for x in last_10 if x == 1) / len(last_10) if last_10 else None)
        top3_rate_10.append(sum(1 for x in last_10 if x <= 3) / len(last_10) if last_10 else None)
        avg_finish_10.append(float(np.mean(last_10)) if last_10 else None)

        last_5_l3 = l3s[-5:]
        avg_last_3f_5.append(float(np.mean(last_5_l3)) if last_5_l3 else None)

        if len(fps) >= 2:
            weights = np.array([0.5 ** i for i in range(len(fps))])[::-1]
            ewm_finish.append(float(np.average(fps, weights=weights)))
        elif fps:
            ewm_finish.append(float(fps[-1]))
        else:
            ewm_finish.append(None)

        days_since.append((current_date - prior[-1][0]).days if prior else None)
        total_races.append(len(prior))

        fp = finish_positions[idx]
        l3 = last_3f_times[idx]
        rt = race_time_secs[idx]
        if fp is not None and fp >= 1:
            history.append((current_date, int(fp), l3, rt))

    cols = [
        *[pl.Series(f"prev{n+1}_finish_pos", prev_finish[n], dtype=pl.Float64) for n in range(5)],
        *[pl.Series(f"prev{n+1}_last_3f", prev_last_3f[n], dtype=pl.Float64) for n in range(5)],
        *[pl.Series(f"prev{n+1}_race_time_sec", prev_time[n], dtype=pl.Float64) for n in range(5)],
        pl.Series("win_rate_10", win_rate_10, dtype=pl.Float64),
        pl.Series("top3_rate_10", top3_rate_10, dtype=pl.Float64),
        pl.Series("avg_finish_10", avg_finish_10, dtype=pl.Float64),
        pl.Series("avg_last_3f_5", avg_last_3f_5, dtype=pl.Float64),
        pl.Series("ewm_finish", ewm_finish, dtype=pl.Float64),
        pl.Series("days_since_last_race", days_since, dtype=pl.Int64),
        pl.Series("total_races", total_races, dtype=pl.Int64),
    ]
    return group.with_columns(cols)
