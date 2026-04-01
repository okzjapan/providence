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

    win_rate_10: list[float | None] = []
    top2_rate_10: list[float | None] = []
    top3_rate_10: list[float | None] = []
    win_rate_30: list[float | None] = []
    top3_rate_30: list[float | None] = []
    avg_finish_10: list[float | None] = []
    avg_finish_30: list[float | None] = []
    momentum: list[float | None] = []
    days_since_last_race: list[int | None] = []
    total_races: list[int] = []

    finished_history: list[tuple[date, int]] = []

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
        momentum.append(_finish_trend(last_5))
        days_since_last_race.append((current_date - prior_history[-1][0]).days if prior_history else None)
        total_races.append(len(prior_history))

        position = finish_positions[idx]
        if position is not None and position >= 1:
            finished_history.append((current_date, int(position)))

    return group.with_columns(
        pl.Series("win_rate_10", win_rate_10),
        pl.Series("top2_rate_10", top2_rate_10),
        pl.Series("top3_rate_10", top3_rate_10),
        pl.Series("win_rate_30", win_rate_30),
        pl.Series("top3_rate_30", top3_rate_30),
        pl.Series("avg_finish_10", avg_finish_10),
        pl.Series("avg_finish_30", avg_finish_30),
        pl.Series("momentum", momentum),
        pl.Series("days_since_last_race", days_since_last_race),
        pl.Series("total_races", total_races),
    )


def _rate(values: Sequence[int], predicate) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if predicate(value)) / len(values))


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _finish_trend(values: Sequence[int]) -> float | None:
    if len(values) < 2:
        return None
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)
