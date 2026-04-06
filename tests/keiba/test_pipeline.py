"""Tests for KeibaFeaturePipeline."""

from datetime import date

import polars as pl
import pytest

from providence.keiba.features.pipeline import KeibaFeaturePipeline


@pytest.fixture()
def pipeline():
    return KeibaFeaturePipeline()


class TestBuildFeatures:
    def test_empty_dataframe(self, pipeline):
        df = pl.DataFrame()
        result = pipeline.build_features(df)
        assert result.is_empty()

    def test_history_end_for_date(self):
        assert KeibaFeaturePipeline.history_end_for_date(date(2024, 6, 15)) == date(2024, 6, 14)


class TestEncategoricals:
    def test_class_code_encoding(self, pipeline):
        df = pl.DataFrame({"class_code": ["05", "A1", "OP", None, "16"]})
        result = pipeline._encode_categoricals(df)
        assert result["class_code"].to_list() == [0, 3, 3, -1, 2]

    def test_unknown_class_code_maps_to_negative(self, pipeline):
        df = pl.DataFrame({"class_code": ["ZZ"]})
        result = pipeline._encode_categoricals(df)
        assert result["class_code"].to_list() == [-1]


class TestAssertNoLeakage:
    def test_passes_with_clean_features(self, pipeline):
        df = pl.DataFrame({
            "idm": [1.0], "distance": [1600],
            "race_date": [date(2024, 1, 1)], "race_number": [1],
        })
        pipeline.assert_no_leakage(df)

    def test_empty_passes(self, pipeline):
        pipeline.assert_no_leakage(pl.DataFrame())


class TestFeatureColumns:
    def test_excludes_known_columns(self, pipeline):
        df = pl.DataFrame({
            "race_id": [1], "idm": [1.0], "distance": [1600],
            "finish_position": [1], "blood_registration_number": ["123"],
        })
        cols = pipeline.feature_columns(df)
        assert "idm" in cols
        assert "distance" in cols
        assert "race_id" not in cols
        assert "finish_position" not in cols
        assert "blood_registration_number" not in cols
