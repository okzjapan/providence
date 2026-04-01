"""Dataset split strategies for train/validation/test."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass(frozen=True)
class SplitRanges:
    warmup_start: date
    warmup_end: date
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date


class SplitStrategy:
    """Month-based split strategy."""

    def auto_split(self, df: pl.DataFrame) -> SplitRanges:
        months = sorted({d.strftime("%Y-%m") for d in df["race_date"].to_list()})
        if len(months) < 19:
            raise ValueError(
                f"データ範囲が短すぎます（{len(months)}ヶ月、最低19ヶ月必要）。"
                "手動分割 (--train-end, --val-end) を使用してください。"
            )

        warmup_end = _first_day_of_month(months[6])
        val_start = _first_day_of_month(months[-12])
        test_start = _first_day_of_month(months[-6])
        dataset_start = _first_day_of_month(months[0])
        latest_end = _first_day_of_next_month(months[-1])
        return SplitRanges(
            warmup_start=dataset_start,
            warmup_end=warmup_end,
            train_start=warmup_end,
            train_end=val_start,
            val_start=val_start,
            val_end=test_start,
            test_start=test_start,
            test_end=latest_end,
        )

    def manual_split(self, df: pl.DataFrame, train_end: date, val_end: date) -> SplitRanges:
        months = sorted({d.strftime("%Y-%m") for d in df["race_date"].to_list()})
        if len(months) < 7:
            raise ValueError("manual_split には最低7ヶ月のデータが必要です。")
        dataset_start = _first_day_of_month(months[0])
        warmup_end = _first_day_of_month(months[6])
        latest_end = _first_day_of_next_month(months[-1])
        train_end_date = train_end + timedelta(days=1)
        val_end_date = val_end + timedelta(days=1)
        if train_end_date <= warmup_end:
            raise ValueError("train_end はウォームアップ終了日より後である必要があります。")
        if val_end_date <= train_end_date:
            raise ValueError("val_end は train_end より後である必要があります。")
        if val_end_date >= latest_end:
            raise ValueError("val_end はデータ末尾より前である必要があります。")
        return SplitRanges(
            warmup_start=dataset_start,
            warmup_end=warmup_end,
            train_start=warmup_end,
            train_end=train_end_date,
            val_start=train_end_date,
            val_end=val_end_date,
            test_start=val_end_date,
            test_end=latest_end,
        )


def apply_split(df: pl.DataFrame, split: SplitRanges) -> dict[str, pl.DataFrame]:
    return {
        "warmup": df.filter((pl.col("race_date") >= split.warmup_start) & (pl.col("race_date") < split.warmup_end)),
        "train": df.filter((pl.col("race_date") >= split.train_start) & (pl.col("race_date") < split.train_end)),
        "val": df.filter((pl.col("race_date") >= split.val_start) & (pl.col("race_date") < split.val_end)),
        "test": df.filter((pl.col("race_date") >= split.test_start) & (pl.col("race_date") < split.test_end)),
    }


def _first_day_of_month(month_str: str) -> date:
    return date.fromisoformat(f"{month_str}-01")


def _first_day_of_next_month(month_str: str) -> date:
    year, month = map(int, month_str.split("-"))
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)
