from datetime import date

import pytest

from providence.model.store import ModelStore


def test_load_for_backtest_resolves_latest_eligible_version(tmp_path, monkeypatch):
    store = ModelStore(base_dir=str(tmp_path / "models"))
    versions = [
        {"version": "v001", "trained_through_date": "2025-01-31"},
        {"version": "v002", "trained_through_date": "2025-03-15"},
        {"version": "v003", "trained_through_date": "2025-06-01"},
    ]
    monkeypatch.setattr(store, "list_versions", lambda: versions)
    monkeypatch.setattr(store, "load", lambda version="latest": (None, {"version": version}))

    _, metadata = store.load_for_backtest(as_of_date=date(2025, 4, 1), mode="walk-forward")
    assert metadata["version"] == "v002"


def test_load_for_backtest_raises_when_no_eligible_model(tmp_path, monkeypatch):
    store = ModelStore(base_dir=str(tmp_path / "models"))
    monkeypatch.setattr(store, "list_versions", lambda: [{"version": "v001", "trained_through_date": "2025-06-01"}])

    with pytest.raises(FileNotFoundError):
        store.load_for_backtest(as_of_date=date(2025, 5, 1), mode="walk-forward")
"""Tests for model store."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from providence.model.store import ModelStore


def _tiny_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    model = lgb.train({"objective": "regression", "metric": "l2", "verbosity": -1}, ds, num_boost_round=3)
    return model


def test_model_store_save_and_load(tmp_path: Path):
    store = ModelStore(base_dir=str(tmp_path / "models"))
    model = _tiny_model()
    version = store.save(model, {"temperature": 1.0, "feature_columns": ["x"]})
    loaded_model, metadata = store.load(version)
    assert version == "v001"
    assert metadata["temperature"] == 1.0
    preds = loaded_model.predict(np.array([[0.5]]))
    assert len(preds) == 1


def test_save_candidate_does_not_promote_latest(tmp_path: Path):
    store = ModelStore(base_dir=str(tmp_path / "models"))
    model = _tiny_model()
    promoted = store.save(model, {"temperature": 1.0, "feature_columns": ["x"]}, version="v001")
    candidate = store.save_candidate(model, {"temperature": 1.0, "feature_columns": ["x"]}, version="v002")
    assert promoted == "v001"
    assert candidate == "v002"
    assert store.latest_version() == "v001"


def test_promote_updates_latest(tmp_path: Path):
    store = ModelStore(base_dir=str(tmp_path / "models"))
    model = _tiny_model()
    store.save(model, {"temperature": 1.0, "feature_columns": ["x"]}, version="v001")
    store.save_candidate(model, {"temperature": 1.0, "feature_columns": ["x"]}, version="v002")
    promoted = store.promote("v002")
    assert promoted == "v002"
    assert store.latest_version() == "v002"
