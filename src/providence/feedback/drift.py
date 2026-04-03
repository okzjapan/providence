"""Drift detection from paper-trade metrics and approximate PSI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from sqlalchemy.orm import Session

from providence.database.repository import Repository
from providence.feedback.psi import approximate_psi_for_frame, load_feature_stats
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.store import ModelStore


@dataclass(frozen=True)
class DriftResult:
    checked_at: date
    warnings: list[str]
    metrics: dict[str, float]
    psi_scores: dict[str, float]

    @property
    def has_warning(self) -> bool:
        return bool(self.warnings)


def detect_drift(
    session: Session,
    *,
    model_version: str,
    evaluation_date: date,
    repository: Repository | None = None,
    loader: DataLoader | None = None,
    pipeline: FeaturePipeline | None = None,
    store: ModelStore | None = None,
) -> DriftResult:
    repository = repository or Repository()
    loader = loader or DataLoader()
    pipeline = pipeline or FeaturePipeline()
    store = store or ModelStore()

    performances = repository.get_recent_model_performance(session, model_version=model_version, limit=12)
    current_4w = next((row for row in performances if row.window == "4w" and row.evaluation_date == evaluation_date), None)
    baseline_4w = [
        row
        for row in performances
        if row.window == "4w" and row.evaluation_date < evaluation_date
    ]

    warnings: list[str] = []
    metrics: dict[str, float] = {}
    if current_4w is not None:
        if current_4w.win_accuracy is not None:
            metrics["win_accuracy"] = float(current_4w.win_accuracy)
        if current_4w.brier_score is not None:
            metrics["brier_score"] = float(current_4w.brier_score)
        if current_4w.roi is not None:
            metrics["roi"] = float(current_4w.roi)
        if baseline_4w:
            baseline_win = _mean([row.win_accuracy for row in baseline_4w])
            baseline_brier = _mean([row.brier_score for row in baseline_4w])
            baseline_roi = _mean([row.roi for row in baseline_4w])
            if current_4w.win_accuracy is not None and baseline_win is not None and current_4w.win_accuracy < baseline_win - 0.05:
                warnings.append("win_accuracy_drop")
            if current_4w.brier_score is not None and baseline_brier is not None and current_4w.brier_score > baseline_brier + 0.02:
                warnings.append("brier_deterioration")
            if current_4w.roi is not None and baseline_roi is not None and current_4w.roi < baseline_roi - 0.10:
                warnings.append("roi_drop")

    psi_scores: dict[str, float] = {}
    try:
        baseline_stats = load_feature_stats(store, model_version)
        raw_df = loader.load_race_dataset(end_date=evaluation_date)
        if not raw_df.is_empty():
            cache_key = FeaturePipeline.cache_key(
                {
                    "purpose": "drift",
                    "rows": len(raw_df),
                    "race_min": raw_df["race_id"].min(),
                    "race_max": raw_df["race_id"].max(),
                    "entry_max": raw_df["race_entry_id"].max(),
                    "date_min": raw_df["race_date"].min(),
                    "date_max": raw_df["race_date"].max(),
                    "end_date": evaluation_date.isoformat(),
                }
            )
            cache_path = Path("data/processed") / f"drift_features_{cache_key}.parquet"
            feature_df = pipeline.build_and_cache(raw_df, str(cache_path))
            recent_df = feature_df.filter(
                pl.col("race_date").is_between(evaluation_date - timedelta(days=27), evaluation_date, closed="both")
            )
            psi_scores = approximate_psi_for_frame(recent_df, baseline_stats)
            if any(score > 0.2 for score in psi_scores.values()):
                warnings.append("psi_drift")
    except FileNotFoundError:
        warnings.append("psi_unavailable")

    return DriftResult(
        checked_at=evaluation_date,
        warnings=sorted(set(warnings)),
        metrics=metrics,
        psi_scores=psi_scores,
    )


def _mean(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
