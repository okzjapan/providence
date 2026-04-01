"""Tests for split strategy."""

from datetime import date

import polars as pl
import pytest

from providence.model.split import SplitStrategy, apply_split


def _df_from_months(start_year=2021, start_month=1, count=24) -> pl.DataFrame:
    dates = []
    year = start_year
    month = start_month
    for _ in range(count):
        dates.append(date(year, month, 1))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return pl.DataFrame({"race_date": dates})


def test_auto_split_requires_minimum_19_months():
    df = _df_from_months(count=18)
    with pytest.raises(ValueError):
        SplitStrategy().auto_split(df)


def test_auto_split_uses_month_boundaries():
    df = _df_from_months(count=24)
    split = SplitStrategy().auto_split(df)
    assert split.warmup_start == date(2021, 1, 1)
    assert split.warmup_end == date(2021, 7, 1)
    assert split.val_start == date(2022, 1, 1)
    assert split.test_start == date(2022, 7, 1)
    assert split.test_end == date(2023, 1, 1)


def test_apply_split_uses_half_open_intervals():
    df = pl.DataFrame({"race_date": [date(2021, 1, 1), date(2021, 7, 1), date(2022, 1, 1), date(2022, 7, 1)]})
    split = SplitStrategy().auto_split(_df_from_months(count=24))
    parts = apply_split(df, split)
    assert len(parts["warmup"]) == 1
    assert len(parts["train"]) == 1
    assert len(parts["val"]) == 1
    assert len(parts["test"]) == 1
