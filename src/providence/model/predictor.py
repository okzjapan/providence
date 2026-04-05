"""Prediction pipeline."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.ensemble import combine_race_scores
from providence.model.store import ModelStore
from providence.probability.plackett_luce import compute_all_ticket_probs
from providence.strategy.types import RaceIndexMap, RacePredictionBundle


def _build_X(features: pl.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: (
                features[column].cast(pl.Int32).fill_null(-1).to_list()
                if column in FeaturePipeline.categorical_columns
                else features[column].cast(pl.Float64).fill_null(float("nan")).to_list()
            )
            for column in feature_columns
        }
    )


class Predictor:
    """End-to-end prediction: features -> scores -> probabilities.

    Transparently handles both single-model and ensemble model versions.
    """

    def __init__(
        self,
        model_store: ModelStore,
        feature_pipeline: FeaturePipeline,
        data_loader: DataLoader,
        *,
        version: str = "latest",
    ) -> None:
        self._is_ensemble = model_store.is_ensemble(version)
        if self._is_ensemble:
            self._models, self._weights, self.metadata = model_store.load_ensemble(version)
            self.model = None
            self.temperature = 1.0
        else:
            self.model, self.metadata = model_store.load(version)
            self._models = None
            self._weights = None
            self.temperature = float(self.metadata["temperature"])

        self.model_version = str(self.metadata["version"])
        self.pipeline = feature_pipeline
        self.loader = data_loader
        self._history: pl.DataFrame | None = None
        self._history_date: date | None = None

    def load_history(self, as_of_date: date) -> None:
        """Load prior-day history only to avoid same-day leakage."""
        cutoff = self.pipeline.history_end_for_date(as_of_date)
        if self._history_date != cutoff:
            self._history = self.loader.load_race_dataset(end_date=cutoff)
            self._history_date = cutoff

    def predict_race(self, race_entries_df: pl.DataFrame) -> RacePredictionBundle:
        if self._history is None:
            raise ValueError("load_history() を先に呼んでください")
        features = self.pipeline.build_features_for_race(race_entries_df, self._history)
        return self.predict_feature_rows(features)

    def predict_races(self, race_entries_df: pl.DataFrame) -> dict[int, RacePredictionBundle]:
        if self._history is None:
            raise ValueError("load_history() を先に呼んでください")
        if race_entries_df.is_empty():
            return {}
        features = self.pipeline.build_features_for_races(race_entries_df, self._history)
        return self.predict_feature_races(features)

    def predict_feature_rows(self, feature_rows: pl.DataFrame) -> RacePredictionBundle:
        if feature_rows.is_empty():
            raise ValueError("feature_rows が空です")
        return self._bundle_from_features(feature_rows.sort("post_position"))

    def predict_feature_races(self, feature_rows: pl.DataFrame) -> dict[int, RacePredictionBundle]:
        if feature_rows.is_empty():
            return {}
        bundles: dict[int, RacePredictionBundle] = {}
        for race_id in feature_rows["race_id"].unique(maintain_order=True).to_list():
            race_features = feature_rows.filter(pl.col("race_id") == race_id).sort("post_position")
            bundle = self._bundle_from_features(race_features)
            bundles[bundle.race_id] = bundle
        return bundles

    def _bundle_from_features(self, features: pl.DataFrame) -> RacePredictionBundle:
        feature_columns = self.metadata["feature_columns"]
        X = _build_X(features, feature_columns)

        if self._is_ensemble and self._models is not None:
            scores, temperature = self._ensemble_predict(X)
        else:
            scores = np.asarray(self.model.predict(X), dtype=float)
            temperature = self.temperature

        index_map = RaceIndexMap(
            index_to_post_position=tuple(int(value) for value in features["post_position"].to_list()),
            index_to_entry_id=tuple(int(value) for value in features["race_entry_id"].to_list()),
        )
        total_races = tuple(
            int(value) if value is not None else 0
            for value in features.get_column("total_races").fill_null(0).to_list()
        )
        return RacePredictionBundle(
            race_id=int(features["race_id"][0]),
            model_version=self.model_version,
            temperature=temperature,
            scores=tuple(float(s) for s in scores),
            index_map=index_map,
            ticket_probs=compute_all_ticket_probs(scores, temperature),
            features_total_races=total_races,
        )

    def _ensemble_predict(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        raw_outputs: dict[str, np.ndarray] = {}
        for key, model in self._models.items():
            raw_outputs[key] = np.asarray(model.predict(X), dtype=float)
        log_scores = combine_race_scores(raw_outputs, self._weights)
        return log_scores, 1.0
