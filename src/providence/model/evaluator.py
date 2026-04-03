"""Evaluation utilities for trained models."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.metrics import brier_score_loss

from providence.features.pipeline import FeaturePipeline
from providence.probability.plackett_luce import compute_win_probs


class Evaluator:
    """Evaluate trained models at race level."""

    def __init__(self, pipeline: FeaturePipeline | None = None) -> None:
        self.pipeline = pipeline or FeaturePipeline()

    def evaluate(self, model: lgb.Booster, test_df: pl.DataFrame, temperature: float = 1.0) -> dict[str, float]:
        feature_columns = self.pipeline.feature_columns(test_df)
        ordered = test_df.sort(["race_date", "race_number", "race_id", "post_position"])
        X = pd.DataFrame(
            {
                column: (
                    ordered[column].cast(pl.Int32).fill_null(-1).to_list()
                    if column in FeaturePipeline.categorical_columns
                    else ordered[column].cast(pl.Float64).fill_null(float("nan")).to_list()
                )
                for column in feature_columns
            }
        )
        scores = np.asarray(model.predict(X), dtype=float)
        with_scores = ordered.with_columns(pl.Series("score", scores))

        race_groups = with_scores.partition_by("race_id", maintain_order=True)
        win_hits = 0
        top3_overlap_total = 0.0
        win_prob_preds: list[float] = []
        win_labels: list[int] = []

        for group in race_groups:
            finish_positions = group["finish_position"].to_list()
            if not finish_positions:
                continue

            actual_winner_idx = _winner_index(finish_positions)
            if actual_winner_idx is None:
                continue

            predicted_order = np.argsort(-group["score"].to_numpy())
            if predicted_order[0] == actual_winner_idx:
                win_hits += 1

            actual_top3 = {idx for idx, pos in enumerate(finish_positions) if pos is not None and 1 <= pos <= 3}
            predicted_top3 = set(predicted_order[: min(3, len(predicted_order))])
            top3_overlap_total += len(actual_top3 & predicted_top3) / 3.0

            win_probs = compute_win_probs(group["score"].to_numpy(), temperature)
            for idx, prob in enumerate(win_probs):
                win_prob_preds.append(float(prob))
                win_labels.append(1 if idx == actual_winner_idx else 0)

        evaluated_race_count = (
            sum(1 for group in race_groups if _winner_index(group["finish_position"].to_list()) is not None) or 1
        )
        baseline = _uniform_brier_baseline(race_groups)
        return {
            "win_accuracy": win_hits / evaluated_race_count,
            "top3_overlap": top3_overlap_total / evaluated_race_count,
            "brier_score": brier_score_loss(win_labels, win_prob_preds) if win_labels else float("nan"),
            "brier_baseline": baseline,
        }

    def feature_importance(self, model: lgb.Booster) -> pl.DataFrame:
        names = model.feature_name()
        return pl.DataFrame(
            {
                "feature": names,
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            }
        ).sort("gain", descending=True)

    def feature_stats(self, df: pl.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float | int | None]]:
        stats: dict[str, dict[str, float | int | None]] = {}
        for column in feature_columns:
            series = df[column]
            is_numeric = series.dtype.is_numeric() and series.null_count() < len(series)
            stats[column] = {
                "null_count": series.null_count(),
                "n_unique": series.n_unique(),
                "mean": float(series.mean()) if is_numeric else None,
                "std": float(series.std()) if is_numeric else None,
                "min": float(series.min()) if is_numeric else None,
                "max": float(series.max()) if is_numeric else None,
                "q10": float(series.quantile(0.10)) if is_numeric else None,
                "q50": float(series.quantile(0.50)) if is_numeric else None,
                "q90": float(series.quantile(0.90)) if is_numeric else None,
            }
        return stats

    def shap_analysis(
        self,
        model: lgb.Booster,
        sample_df: pl.DataFrame,
        n_samples: int = 1000,
    ) -> pl.DataFrame:
        """Return mean absolute SHAP values on a sampled subset."""
        if sample_df.is_empty():
            return pl.DataFrame({"feature": [], "mean_abs_shap": []})

        feature_columns = self.pipeline.feature_columns(sample_df)
        sample = sample_df.head(min(n_samples, len(sample_df)))
        X = pd.DataFrame(
            {
                column: (
                    sample[column].cast(pl.Int32).fill_null(-1).to_list()
                    if column in FeaturePipeline.categorical_columns
                    else sample[column].cast(pl.Float64).fill_null(float("nan")).to_list()
                )
                for column in feature_columns
            }
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        return pl.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs}).sort(
            "mean_abs_shap", descending=True
        )


def _winner_index(finish_positions: list[int | None]) -> int | None:
    for idx, pos in enumerate(finish_positions):
        if pos == 1:
            return idx
    return None


def _uniform_brier_baseline(race_groups: list[pl.DataFrame]) -> float:
    preds: list[float] = []
    labels: list[float] = []
    for group in race_groups:
        n = len(group)
        finish_positions = group["finish_position"].to_list()
        winner_idx = _winner_index(finish_positions)
        if winner_idx is None:
            continue
        probs = np.full(n, 1.0 / n)
        race_labels = np.zeros(n)
        race_labels[winner_idx] = 1.0
        preds.extend(probs.tolist())
        labels.extend(race_labels.tolist())
    return brier_score_loss(labels, preds) if labels else float("nan")
