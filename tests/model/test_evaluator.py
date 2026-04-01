"""Tests for evaluator utilities."""

from datetime import date

import polars as pl

from providence.features.pipeline import FeaturePipeline
from providence.model.evaluator import Evaluator
from providence.model.trainer import Trainer


def _raw_df() -> pl.DataFrame:
    rows = []
    race_id = 1
    for race_date in [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), date(2021, 1, 4)]:
        for rider_id, pos, trial, finish in [
            (1, 1, 3.30, 1),
            (2, 2, 3.35, 2),
            (3, 3, 3.40, 3),
        ]:
            rows.append(
                {
                    "race_id": race_id,
                    "race_entry_id": race_id * 10 + rider_id,
                    "race_date": race_date,
                    "race_number": 1,
                    "track_id": 4,
                    "track_condition": "良",
                    "weather": "晴",
                    "temperature": 15.0,
                    "humidity": 40.0,
                    "track_temperature": 20.0,
                    "grade": "普通",
                    "distance": 3100,
                    "race_status": "正常",
                    "rider_id": rider_id,
                    "rider_registration_number": str(rider_id),
                    "generation": 30,
                    "birth_year": 1990,
                    "home_track_id": 4,
                    "post_position": pos,
                    "handicap_meters": (pos - 1) * 10,
                    "trial_time": trial,
                    "avg_trial_time": None,
                    "trial_deviation": 0.08 + pos * 0.01,
                    "race_score": None,
                    "entry_status": "出走",
                    "finish_position": finish,
                    "race_time": 3.4 + pos * 0.01,
                    "start_timing": 0.1 + pos * 0.01,
                    "accident_code": None,
                    "row_key": f"{race_id}-{rider_id}",
                }
            )
        race_id += 1
    return pl.DataFrame(rows)


def test_evaluator_returns_expected_metric_keys():
    pipeline = FeaturePipeline()
    features = pipeline.build_features(_raw_df())
    train_df = features.filter(pl.col("race_date") < date(2021, 1, 4))
    test_df = features.filter(pl.col("race_date") == date(2021, 1, 4))
    model = Trainer(pipeline=pipeline).train_lambdarank(train_df, test_df).model
    metrics = Evaluator(pipeline=pipeline).evaluate(model, test_df, temperature=1.0)
    assert set(metrics) == {"win_accuracy", "top3_overlap", "brier_score", "brier_baseline"}
