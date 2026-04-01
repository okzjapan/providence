"""Tests for predictor pipeline."""

from datetime import date
from pathlib import Path

import polars as pl

from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
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


class DummyLoader(DataLoader):
    def __init__(self, history: pl.DataFrame) -> None:
        self._history = history

    def load_race_dataset(self, start_date=None, end_date=None) -> pl.DataFrame:  # noqa: ARG002
        return self._history


def test_predictor_returns_all_ticket_types(tmp_path: Path):
    pipeline = FeaturePipeline()
    raw = _raw_df()
    features = pipeline.build_features(raw)
    train_df = features.filter(pl.col("race_date") < date(2021, 1, 4))
    history_raw = raw.filter(pl.col("race_date") < date(2021, 1, 4))
    target_df = raw.filter(pl.col("race_date") == date(2021, 1, 4))
    artifacts = Trainer(pipeline=pipeline).train_lambdarank(train_df, train_df)

    store = ModelStore(base_dir=str(tmp_path / "models"))
    store.save(
        artifacts.model,
        {"temperature": 1.0, "feature_columns": artifacts.feature_columns},
        version="v001",
    )

    predictor = Predictor(store, pipeline, DummyLoader(history_raw))
    predictor.load_history(date(2021, 1, 4))
    probs = predictor.predict_race(target_df)
    assert set(probs) == {"win", "exacta", "quinella", "trifecta", "trio", "wide"}
