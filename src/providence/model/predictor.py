"""Prediction pipeline."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import polars as pl

from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.store import ModelStore
from providence.probability.plackett_luce import compute_all_ticket_probs


class Predictor:
    """End-to-end prediction: features -> scores -> probabilities."""

    def __init__(
        self,
        model_store: ModelStore,
        feature_pipeline: FeaturePipeline,
        data_loader: DataLoader,
    ) -> None:
        self.model, self.metadata = model_store.load("latest")
        self.temperature = float(self.metadata["temperature"])
        self.pipeline = feature_pipeline
        self.loader = data_loader
        self._history: pl.DataFrame | None = None
        self._history_date: date | None = None

    def load_history(self, as_of_date: date) -> None:
        """Load prior-day history only to avoid same-day leakage."""
        cutoff = as_of_date - timedelta(days=1)
        if self._history_date != cutoff:
            self._history = self.loader.load_race_dataset(end_date=cutoff)
            self._history_date = cutoff

    def predict_race(self, race_entries_df: pl.DataFrame) -> dict[str, dict]:
        if self._history is None:
            raise ValueError("load_history() を先に呼んでください")
        features = self.pipeline.build_features_for_race(race_entries_df, self._history)
        feature_columns = self.metadata["feature_columns"]
        X = pd.DataFrame(
            {
                column: (
                    features[column].cast(pl.Int32).fill_null(-1).to_list()
                    if column in FeaturePipeline.categorical_columns
                    else features[column].cast(pl.Float64).fill_null(float("nan")).to_list()
                )
                for column in feature_columns
            }
        )
        scores = self.model.predict(X)
        return compute_all_ticket_probs(scores, self.temperature)
