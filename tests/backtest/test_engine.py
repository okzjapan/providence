from datetime import date, time

import polars as pl

from providence.backtest.engine import BacktestEngine
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.store import ModelStore
from providence.strategy.types import EvaluationMode


class DummyLoader(DataLoader):
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def load_race_dataset(self, start_date=None, end_date=None) -> pl.DataFrame:  # noqa: ARG002
        return self._df


def test_walk_forward_skips_dates_without_eligible_model(monkeypatch):
    df = pl.DataFrame(
        {
            "race_id": [1],
            "race_date": [date(2025, 1, 1)],
            "track_id": [4],
            "race_number": [1],
            "track_condition": ["良"],
            "weather": ["晴"],
            "temperature": [15.0],
            "humidity": [40.0],
            "track_temperature": [20.0],
            "grade": ["普通"],
            "distance": [3100],
            "race_status": ["正常"],
            "race_entry_id": [10],
            "rider_id": [1],
            "post_position": [1],
            "handicap_meters": [0],
            "trial_time": [3.3],
            "avg_trial_time": [None],
            "trial_deviation": [0.1],
            "race_score": [None],
            "entry_status": ["出走"],
            "finish_position": [1],
            "race_time": [3.4],
            "start_timing": [0.1],
            "accident_code": [None],
            "rider_registration_number": ["1"],
            "generation": [30],
            "birth_year": [1990],
            "home_track_id": [4],
            "row_key": ["1-1"],
        }
    )
    model_store = ModelStore(base_dir="data/models")
    monkeypatch.setattr(model_store, "load_for_backtest", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    engine = BacktestEngine(loader=DummyLoader(df), pipeline=FeaturePipeline(), model_store=model_store)
    results = engine.run(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        judgment_clock=time(10, 0),
        evaluation_mode=EvaluationMode.WALK_FORWARD,
    )
    assert results == []
