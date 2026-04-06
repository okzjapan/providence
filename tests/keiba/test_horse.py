"""Tests for keiba horse features."""

from datetime import date

import polars as pl

from providence.keiba.features.horse import _distance_category, add_horse_features


class TestDistanceCategory:
    def test_sprint(self):
        assert _distance_category(1200) == "sprint"

    def test_mile(self):
        assert _distance_category(1600) == "mile"

    def test_middle(self):
        assert _distance_category(2000) == "middle"

    def test_long(self):
        assert _distance_category(2400) == "long"

    def test_none(self):
        assert _distance_category(None) is None


def _make_df(rows):
    return pl.DataFrame(rows).with_columns(pl.col("race_date").cast(pl.Date))


class TestAddHorseFeatures:
    def test_empty(self):
        result = add_horse_features(pl.DataFrame())
        assert result.is_empty()

    def test_first_surface_detection(self):
        df = _make_df([
            {"blood_registration_number": "H001", "race_date": date(2024, 1, 1), "race_number": 1, "post_position": 1,
             "surface_code": 1, "distance": 1600, "racecourse_id": 5, "finish_position": 1},
            {"blood_registration_number": "H001", "race_date": date(2024, 2, 1), "race_number": 1, "post_position": 1,
             "surface_code": 2, "distance": 1600, "racecourse_id": 5, "finish_position": 2},
        ])
        result = add_horse_features(df)
        assert result["is_first_surface"][0] is None
        assert result["is_first_surface"][1] is True

    def test_distance_change(self):
        df = _make_df([
            {"blood_registration_number": "H001", "race_date": date(2024, 1, 1), "race_number": 1, "post_position": 1,
             "surface_code": 1, "distance": 1600, "racecourse_id": 5, "finish_position": 1},
            {"blood_registration_number": "H001", "race_date": date(2024, 2, 1), "race_number": 1, "post_position": 1,
             "surface_code": 1, "distance": 2000, "racecourse_id": 5, "finish_position": 2},
        ])
        result = add_horse_features(df)
        assert result["distance_change"][0] is None
        assert result["distance_change"][1] == 400
