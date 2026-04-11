"""Rider performance features."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl


def add_rider_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rider-level historical performance features."""
    if df.is_empty():
        return df

    return (
        df.sort(["rider_id", "race_date", "race_number", "post_position"])
        .group_by("rider_id", maintain_order=True)
        .map_groups(_add_group_features)
    )


def _add_group_features(group: pl.DataFrame) -> pl.DataFrame:
    race_dates: Sequence[date] = group["race_date"].to_list()
    finish_positions = group["finish_position"].to_list()
    start_timings = group["start_timing"].to_list() if "start_timing" in group.columns else [None] * len(race_dates)

    win_rate_10: list[float | None] = []
    top2_rate_10: list[float | None] = []
    top3_rate_10: list[float | None] = []
    win_rate_30: list[float | None] = []
    top3_rate_30: list[float | None] = []
    avg_finish_10: list[float | None] = []
    avg_finish_30: list[float | None] = []
    finish_std_10: list[float | None] = []
    momentum: list[float | None] = []
    days_since_last_race: list[int | None] = []
    total_races: list[int] = []
    avg_st_10: list[float | None] = []
    avg_st_30: list[float | None] = []
    st_consistency: list[float | None] = []

    win_rate_all: list[float | None] = []
    top3_rate_all: list[float | None] = []
    avg_finish_all: list[float | None] = []
    recent_vs_career_wr: list[float | None] = []
    recent_vs_career_finish: list[float | None] = []
    days_last_win: list[int | None] = []
    days_last_top3: list[int | None] = []
    races_since_win: list[int | None] = []
    accident_rate: list[float | None] = []

    finished_history: list[tuple[date, int]] = []
    st_history: list[float] = []

    for idx, current_date in enumerate(race_dates):
        prior_history = [(d, p) for d, p in finished_history if d < current_date]
        positions_only = [p for _, p in prior_history]
        last_10 = positions_only[-10:]
        last_30 = positions_only[-30:]
        last_5 = positions_only[-5:]

        win_rate_10.append(_rate(last_10, lambda x: x == 1))
        top2_rate_10.append(_rate(last_10, lambda x: x <= 2))
        top3_rate_10.append(_rate(last_10, lambda x: x <= 3))
        win_rate_30.append(_rate(last_30, lambda x: x == 1))
        top3_rate_30.append(_rate(last_30, lambda x: x <= 3))
        avg_finish_10.append(_mean(last_10))
        avg_finish_30.append(_mean(last_30))
        finish_std_10.append(_std(last_10))
        momentum.append(_finish_trend(last_5))
        days_since_last_race.append((current_date - prior_history[-1][0]).days if prior_history else None)
        total_races.append(len(prior_history))

        st_last_10 = st_history[-10:]
        st_last_30 = st_history[-30:]
        avg_st_10.append(_mean_float(st_last_10))
        avg_st_30.append(_mean_float(st_last_30))
        st_consistency.append(_std_float(st_last_10))

        career_wr = _rate(positions_only, lambda x: x == 1) if positions_only else None
        career_t3r = _rate(positions_only, lambda x: x <= 3) if positions_only else None
        career_avg_fin = _mean(positions_only) if positions_only else None
        win_rate_all.append(career_wr)
        top3_rate_all.append(career_t3r)
        avg_finish_all.append(career_avg_fin)
        wr10 = win_rate_10[-1]
        recent_vs_career_wr.append(float(wr10 - career_wr) if wr10 is not None and career_wr is not None else None)
        af10 = avg_finish_10[-1]
        recent_vs_career_finish.append(float(career_avg_fin - af10) if af10 is not None and career_avg_fin is not None else None)

        days_last_win.append(_days_since_condition(prior_history, current_date, lambda p: p == 1))
        days_last_top3.append(_days_since_condition(prior_history, current_date, lambda p: p <= 3))
        races_since_win.append(_races_since_condition(positions_only, lambda p: p == 1))

        accident_codes = group["accident_code"].to_list() if "accident_code" in group.columns else []
        if accident_codes:
            prior_accidents = [accident_codes[j] for j in range(len(finished_history)) if j < idx]
            last30_acc = prior_accidents[-30:]
            accident_rate.append(float(sum(1 for a in last30_acc if a is not None and a != "") / len(last30_acc)) if last30_acc else None)
        else:
            accident_rate.append(None)

        position = finish_positions[idx]
        if position is not None and position >= 1:
            finished_history.append((current_date, int(position)))

        st_val = start_timings[idx]
        if st_val is not None and st_val > 0:
            st_history.append(float(st_val))

    return group.with_columns(
        pl.Series("win_rate_10", win_rate_10),
        pl.Series("top2_rate_10", top2_rate_10),
        pl.Series("top3_rate_10", top3_rate_10),
        pl.Series("win_rate_30", win_rate_30),
        pl.Series("top3_rate_30", top3_rate_30),
        pl.Series("avg_finish_10", avg_finish_10),
        pl.Series("avg_finish_30", avg_finish_30),
        pl.Series("finish_std_10", finish_std_10),
        pl.Series("momentum", momentum),
        pl.Series("days_since_last_race", days_since_last_race),
        pl.Series("total_races", total_races),
        pl.Series("avg_start_timing_10", avg_st_10),
        pl.Series("avg_start_timing_30", avg_st_30),
        pl.Series("start_timing_consistency", st_consistency),
        pl.Series("win_rate_all", win_rate_all),
        pl.Series("top3_rate_all", top3_rate_all),
        pl.Series("avg_finish_all", avg_finish_all),
        pl.Series("recent_vs_career_wr", recent_vs_career_wr),
        pl.Series("recent_vs_career_finish", recent_vs_career_finish),
        pl.Series("days_since_last_win", days_last_win),
        pl.Series("days_since_last_top3", days_last_top3),
        pl.Series("races_since_last_win", races_since_win),
        pl.Series("accident_rate_30", accident_rate),
    )


def _rate(values: Sequence[int], predicate) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if predicate(value)) / len(values))


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std(values: Sequence[int]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def _mean_float(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std_float(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def _finish_trend(values: Sequence[int]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def _days_since_condition(history: list[tuple[date, int]], current: date, pred) -> int | None:
    for d, p in reversed(history):
        if pred(p):
            return (current - d).days
    return None


def _races_since_condition(positions: list[int], pred) -> int | None:
    for i, p in enumerate(reversed(positions)):
        if pred(p):
            return i
    return len(positions) if positions else None
