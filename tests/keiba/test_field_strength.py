"""Tests for keiba field strength features."""

import polars as pl
import pytest

from providence.keiba.features.field_strength import add_field_strength_features


class TestAddFieldStrengthFeatures:
    def test_empty(self):
        result = add_field_strength_features(pl.DataFrame())
        assert result.is_empty()

    def test_basic_calculation(self):
        df = pl.DataFrame({
            "race_id": [1, 1, 1],
            "idm": [30.0, 40.0, 50.0],
        })
        result = add_field_strength_features(df)
        assert result["field_avg_idm"].to_list() == [40.0, 40.0, 40.0]
        assert result["field_max_idm"].to_list() == [50.0, 50.0, 50.0]
        assert result["gap_to_top"][0] == pytest.approx(20.0)
        assert result["gap_to_top"][2] == pytest.approx(0.0)

    def test_relative_idm(self):
        df = pl.DataFrame({
            "race_id": [1, 1],
            "idm": [20.0, 40.0],
        })
        result = add_field_strength_features(df)
        assert result["relative_idm"][0] == pytest.approx(20.0 / 30.0)
        assert result["relative_idm"][1] == pytest.approx(40.0 / 30.0)

    def test_multiple_races(self):
        df = pl.DataFrame({
            "race_id": [1, 1, 2, 2],
            "idm": [30.0, 40.0, 50.0, 60.0],
        })
        result = add_field_strength_features(df)
        assert result["field_avg_idm"][0] == pytest.approx(35.0)
        assert result["field_avg_idm"][2] == pytest.approx(55.0)
