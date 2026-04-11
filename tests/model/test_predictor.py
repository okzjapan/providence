"""Tests for predictor pipeline."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor, _blend_pair_ticket_probs
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
    bundle = predictor.predict_race(target_df)
    assert set(bundle.ticket_probs) == {"win", "place", "exacta", "quinella", "trifecta", "trio", "wide"}
    assert bundle.model_version == "v001"
    assert bundle.index_map.index_to_post_position == (1, 2, 3)


def test_predictor_ensemble_returns_all_ticket_types(tmp_path: Path):
    pipeline = FeaturePipeline()
    raw = _raw_df()
    features = pipeline.build_features(raw)
    train_df = features.filter(pl.col("race_date") < date(2021, 1, 4))
    history_raw = raw.filter(pl.col("race_date") < date(2021, 1, 4))
    target_df = raw.filter(pl.col("race_date") == date(2021, 1, 4))

    t = Trainer(pipeline=pipeline)
    rank_art = t.train_lambdarank(train_df, train_df)
    top2_art = t.train_binary_top2(train_df, train_df)
    win_art = t.train_binary_win(train_df, train_df)
    huber_art = t.train_huber(train_df, train_df)

    store = ModelStore(base_dir=str(tmp_path / "models"))
    models = {
        "lambdarank": rank_art.model,
        "binary_top2": top2_art.model,
        "binary_win": win_art.model,
        "huber": huber_art.model,
    }
    weights = {"lambdarank": 0.4, "binary_top2": 0.3, "binary_win": 0.15, "huber": 0.15}
    store.save_ensemble(
        models,
        weights,
        {"temperature": 1.0, "feature_columns": rank_art.feature_columns},
        version="e001",
    )

    predictor = Predictor(store, pipeline, DummyLoader(history_raw), version="e001")
    predictor.load_history(date(2021, 1, 4))
    bundle = predictor.predict_race(target_df)
    assert set(bundle.ticket_probs) == {"win", "place", "exacta", "quinella", "trifecta", "trio", "wide"}
    assert bundle.model_version == "e001"
    assert bundle.temperature == 1.0


def test_predictor_predict_races_batches_same_day(tmp_path: Path):
    pipeline = FeaturePipeline()
    raw = _raw_df().with_columns(
        pl.when(pl.col("race_date") == date(2021, 1, 4))
        .then(pl.col("race_id") + 10)
        .otherwise(pl.col("race_id"))
        .alias("race_id"),
        pl.when(pl.col("race_date") == date(2021, 1, 4))
        .then(2)
        .otherwise(pl.col("race_number"))
        .alias("race_number"),
    )
    second_race = raw.filter(pl.col("race_date") == date(2021, 1, 3)).with_columns(
        pl.lit(date(2021, 1, 4)).alias("race_date"),
        pl.lit(1).alias("race_number"),
        (pl.col("race_id") + 20).alias("race_id"),
        (pl.col("race_entry_id") + 200).alias("race_entry_id"),
    )
    raw = pl.concat([raw, second_race], how="vertical_relaxed")
    features = pipeline.build_features(raw)
    train_df = features.filter(pl.col("race_date") < date(2021, 1, 4))
    history_raw = raw.filter(pl.col("race_date") < date(2021, 1, 4))
    target_df = raw.filter(pl.col("race_date") == date(2021, 1, 4)).sort(["race_number", "post_position"])
    artifacts = Trainer(pipeline=pipeline).train_lambdarank(train_df, train_df)

    store = ModelStore(base_dir=str(tmp_path / "models"))
    store.save(
        artifacts.model,
        {"temperature": 1.0, "feature_columns": artifacts.feature_columns},
        version="v001",
    )

    predictor = Predictor(store, pipeline, DummyLoader(history_raw))
    predictor.load_history(date(2021, 1, 4))
    bundles = predictor.predict_races(target_df)
    assert len(bundles) == 2
    assert all(set(bundle.ticket_probs) == {"win", "place", "exacta", "quinella", "trifecta", "trio", "wide"} for bundle in bundles.values())


def test_blend_pair_ticket_probs_only_changes_exacta_and_quinella():
    main = {
        "win": {0: 0.6, 1: 0.3, 2: 0.1},
        "exacta": {(0, 1): 0.3, (1, 0): 0.1, (0, 2): 0.2, (2, 0): 0.05, (1, 2): 0.2, (2, 1): 0.15},
        "quinella": {(0, 1): 0.4, (0, 2): 0.25, (1, 2): 0.35},
        "trifecta": {(0, 1, 2): 0.4},
        "trio": {(0, 1, 2): 0.5},
        "wide": {(0, 1): 0.6, (0, 2): 0.4, (1, 2): 0.3},
    }
    aux_strengths = np.array([0.1, 0.8, 0.1])
    blended = _blend_pair_ticket_probs(main, aux_strengths, 0.3)
    assert blended["win"] == main["win"]
    assert blended["trifecta"] == main["trifecta"]
    assert blended["trio"] == main["trio"]
    assert blended["wide"] == main["wide"]
    assert blended["exacta"] != main["exacta"]
    assert blended["quinella"] != main["quinella"]
