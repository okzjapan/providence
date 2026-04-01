"""Tests for feature pipeline behaviour."""

from datetime import date

import polars as pl

from providence.features.pipeline import FeaturePipeline


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_id": [1, 2, 3],
            "race_entry_id": [11, 21, 31],
            "race_date": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 2)],
            "race_number": [1, 2, 1],
            "track_id": [4, 4, 4],
            "track_condition": ["良", "良", "湿"],
            "weather": ["晴", "晴", "雨"],
            "temperature": [10.0, 10.0, 8.0],
            "humidity": [30.0, 30.0, 80.0],
            "track_temperature": [20.0, 20.0, 12.0],
            "grade": ["普通", "普通", "普通"],
            "distance": [3100, 3100, 3100],
            "race_status": ["正常", "正常", "正常"],
            "rider_id": [100, 100, 100],
            "rider_registration_number": ["100", "100", "100"],
            "generation": [30, 30, 30],
            "birth_year": [1990, 1990, 1990],
            "home_track_id": [4, 4, 4],
            "post_position": [1, 1, 1],
            "handicap_meters": [0, 10, 20],
            "trial_time": [3.40, 3.35, 3.30],
            "avg_trial_time": [None, None, None],
            "trial_deviation": [0.09, 0.08, 0.07],
            "race_score": [None, None, None],
            "entry_status": ["出走", "出走", "出走"],
            "finish_position": [2, 1, 3],
            "race_time": [3.50, 3.40, 3.60],
            "start_timing": [0.10, 0.11, 0.12],
            "accident_code": [None, None, None],
            "row_key": ["a", "b", "c"],
        }
    )


def test_build_features_excludes_same_day_previous_race_for_batch_mode():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_sample_df())

    second_race = features.filter(pl.col("race_id") == 2).row(0, named=True)
    assert second_race["days_since_last_race"] is None
    assert second_race["total_races"] == 0
    assert second_race["avg_finish_10"] is None
    assert second_race["win_rate_10"] is None


def test_build_features_for_race_only_returns_target_race():
    pipeline = FeaturePipeline()
    raw = _sample_df()
    history = raw.filter(pl.col("race_date") < date(2021, 1, 2))
    target = raw.filter(pl.col("race_id") == 3)
    features = pipeline.build_features_for_race(target, history)
    assert features["race_id"].unique().to_list() == [3]
    assert len(features) == 1


def test_encode_categoricals_is_stable_and_integer():
    pipeline = FeaturePipeline()
    encoded = pipeline._encode_categoricals(_sample_df())  # noqa: SLF001
    for column in FeaturePipeline.categorical_columns:
        assert encoded[column].dtype in (pl.Int32, pl.Int64)
