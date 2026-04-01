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
