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
        if column in encoded.columns:
            assert encoded[column].dtype in (pl.Int32, pl.Int64)


def _multi_race_df() -> pl.DataFrame:
    """Dataset with enough history for trial deviation computation (>= 5 good-weather races)."""
    n = 8
    return pl.DataFrame(
        {
            "race_id": list(range(1, n + 1)),
            "race_entry_id": list(range(11, 11 + n)),
            "race_date": [date(2021, 1, d) for d in range(1, n + 1)],
            "race_number": [1] * n,
            "track_id": [4] * n,
            "track_condition": ["良"] * 6 + ["湿", "良"],
            "weather": ["晴"] * n,
            "temperature": [10.0] * n,
            "humidity": [30.0] * n,
            "track_temperature": [20.0] * n,
            "grade": ["普通"] * n,
            "distance": [3100] * n,
            "race_status": ["正常"] * n,
            "rider_id": [100] * n,
            "rider_registration_number": ["100"] * n,
            "generation": [30] * n,
            "birth_year": [1990] * n,
            "home_track_id": [4] * n,
            "post_position": [1] * n,
            "handicap_meters": [10] * n,
            "trial_time": [3.40, 3.38, 3.42, 3.39, 3.41, 3.37, 3.35, 3.36],
            "avg_trial_time": [None] * n,
            "trial_deviation": [0.08] * n,
            "race_score": [None] * n,
            "entry_status": ["出走"] * n,
            "finish_position": [2, 1, 3, 1, 2, 1, 3, 1],
            "race_time": [3.50, 3.45, 3.52, 3.47, 3.51, 3.44, 3.55, 3.43],
            "start_timing": [0.10] * n,
            "accident_code": [None] * n,
            "row_key": [f"r{i}" for i in range(1, n + 1)],
        }
    )


def test_computed_trial_deviation_uses_only_past_good_weather():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())

    row6 = features.filter(pl.col("race_id") == 6).row(0, named=True)
    assert row6["computed_trial_deviation"] is not None, "5走の良走路履歴があるのでNoneではない"
    expected = (
        (3.50 - 3.40) + (3.45 - 3.38) + (3.52 - 3.42) + (3.47 - 3.39) + (3.51 - 3.41)
    ) / 5
    assert abs(row6["computed_trial_deviation"] - expected) < 1e-6

    row7 = features.filter(pl.col("race_id") == 7).row(0, named=True)
    assert row7["computed_trial_deviation"] is not None
    expected_7 = (
        (3.50 - 3.40) + (3.45 - 3.38) + (3.52 - 3.42) + (3.47 - 3.39) + (3.51 - 3.41) + (3.44 - 3.37)
    ) / 6
    assert abs(row7["computed_trial_deviation"] - expected_7) < 1e-6


def test_computed_trial_deviation_none_with_insufficient_history():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())

    row3 = features.filter(pl.col("race_id") == 3).row(0, named=True)
    assert row3["computed_trial_deviation"] is None, "2走の良走路履歴のみなので None"


def test_computed_trial_deviation_excludes_wet_races():
    """Race 7 is on wet track (湿) — its deviation must NOT enter the history for race 8."""
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())

    row8 = features.filter(pl.col("race_id") == 8).row(0, named=True)
    assert row8["computed_trial_deviation"] is not None
    good_deviations = [
        3.50 - 3.40, 3.45 - 3.38, 3.52 - 3.42,
        3.47 - 3.39, 3.51 - 3.41, 3.44 - 3.37,
    ]
    expected = sum(good_deviations) / len(good_deviations)
    assert abs(row8["computed_trial_deviation"] - expected) < 1e-6


def test_predicted_race_time_and_rank_present():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())
    assert "predicted_race_time" in features.columns
    assert "predicted_race_time_rank" in features.columns
    assert "trial_time_zscore" in features.columns

    row6 = features.filter(pl.col("race_id") == 6).row(0, named=True)
    if row6["computed_trial_deviation"] is not None and row6["trial_time"] is not None:
        expected_prt = row6["trial_time"] + row6["computed_trial_deviation"]
        assert abs(row6["predicted_race_time"] - expected_prt) < 1e-6


def test_predicted_vs_handicap_present():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())
    assert "predicted_vs_handicap" in features.columns


def test_no_leakage_in_trial_deviation():
    """computed_trial_deviation must not use current or future race_time."""
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_multi_race_df())

    row1 = features.filter(pl.col("race_id") == 1).row(0, named=True)
    assert row1["computed_trial_deviation"] is None, "初走なので None"

    row2 = features.filter(pl.col("race_id") == 2).row(0, named=True)
    assert row2["computed_trial_deviation"] is None, "1走の履歴のみなので None (最低5走必要)"
