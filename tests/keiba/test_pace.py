"""Tests for keiba pace features."""

from datetime import date

import polars as pl

from providence.keiba.features.pace import add_pace_features


def _make_df(rows):
    return pl.DataFrame(rows).with_columns(pl.col("race_date").cast(pl.Date))


class TestAddPaceFeatures:
    def test_empty(self):
        result = add_pace_features(pl.DataFrame())
        assert result.is_empty()

    def test_position_score_calculation(self):
        df = _make_df([
            {"blood_registration_number": "H001", "race_date": date(2024, 1, 1), "race_number": 1, "post_position": 1,
             "corner_4_pos": 1, "num_runners": 16},
            {"blood_registration_number": "H001", "race_date": date(2024, 2, 1), "race_number": 1, "post_position": 1,
             "corner_4_pos": 8, "num_runners": 16},
            {"blood_registration_number": "H001", "race_date": date(2024, 3, 1), "race_number": 1, "post_position": 1,
             "corner_4_pos": 3, "num_runners": 16},
        ])
        result = add_pace_features(df)
        assert result["avg_position_score"][0] is None
        assert result["avg_position_score"][1] is not None
        assert 0.0 <= result["avg_position_score"][1] <= 1.0

    def test_running_style_front_runner(self):
        df = _make_df([
            {"blood_registration_number": "H001", "race_date": date(2024, 1, i), "race_number": 1, "post_position": 1,
             "corner_4_pos": 1, "num_runners": 16}
            for i in range(1, 7)
        ])
        result = add_pace_features(df)
        last_style = result["primary_running_style"][-1]
        assert last_style == 1
