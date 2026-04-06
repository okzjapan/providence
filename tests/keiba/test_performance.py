"""Tests for keiba performance features."""

from datetime import date

import polars as pl

from providence.keiba.features.performance import add_performance_features


def _make_df(rows):
    return pl.DataFrame(rows).with_columns(
        pl.col("race_date").cast(pl.Date),
    )


class TestAddPerformanceFeatures:
    def test_empty(self):
        result = add_performance_features(pl.DataFrame())
        assert result.is_empty()

    def test_prev_finish_positions(self):
        df = _make_df(
            [
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 1, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 3,
                    "last_3f_time": 35.0,
                    "race_time_sec": 95.0,
                },
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 2, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 1,
                    "last_3f_time": 34.0,
                    "race_time_sec": 94.0,
                },
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 3, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 2,
                    "last_3f_time": 34.5,
                    "race_time_sec": 94.5,
                },
            ]
        )
        result = add_performance_features(df)

        assert result["prev1_finish_pos"][0] is None
        assert result["prev1_finish_pos"][1] == 3
        assert result["prev1_finish_pos"][2] == 1
        assert result["prev2_finish_pos"][2] == 3

    def test_no_future_leakage(self):
        df = _make_df(
            [
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 1, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 1,
                    "last_3f_time": 34.0,
                    "race_time_sec": 94.0,
                },
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 2, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 5,
                    "last_3f_time": 36.0,
                    "race_time_sec": 97.0,
                },
            ]
        )
        result = add_performance_features(df)
        assert result["win_rate_10"][0] is None
        assert result["total_races"][0] == 0

    def test_days_since_last_race(self):
        df = _make_df(
            [
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 1, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 1,
                    "last_3f_time": 34.0,
                    "race_time_sec": 94.0,
                },
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 2, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 2,
                    "last_3f_time": 35.0,
                    "race_time_sec": 95.0,
                },
            ]
        )
        result = add_performance_features(df)
        assert result["days_since_last_race"][0] is None
        assert result["days_since_last_race"][1] == 31

    def test_multiple_horses(self):
        df = _make_df(
            [
                {
                    "blood_registration_number": "H001",
                    "race_date": date(2024, 1, 1),
                    "race_number": 1,
                    "post_position": 1,
                    "finish_position": 1,
                    "last_3f_time": 34.0,
                    "race_time_sec": 94.0,
                },
                {
                    "blood_registration_number": "H002",
                    "race_date": date(2024, 1, 1),
                    "race_number": 1,
                    "post_position": 2,
                    "finish_position": 2,
                    "last_3f_time": 35.0,
                    "race_time_sec": 95.0,
                },
            ]
        )
        result = add_performance_features(df)
        assert result["total_races"][0] == 0
        assert result["total_races"][1] == 0
