"""Retrain workflow for candidate model creation and optional promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import sklearn

from providence.backtest.engine import BacktestEngine
from providence.backtest.metrics import summarize_backtest
from providence.config import DEFAULT_BANKROLL_JPY
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.evaluator import Evaluator
from providence.model.split import SplitStrategy, apply_split
from providence.model.store import ModelStore
from providence.model.trainer import Trainer
from providence.probability.calibration import TemperatureScaler
from providence.strategy.types import EvaluationMode


@dataclass(frozen=True)
class RetrainResult:
    candidate_version: str
    baseline_version: str
    promoted_version: str | None
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    backtest_candidate_roi: float | None
    backtest_baseline_roi: float | None


def run_retrain_workflow(
    *,
    compare_with: str = "latest",
    compare_window_days: int = 28,
    n_trials: int = 30,
    optimize: bool = False,
    shap_samples: int = 1000,
    rebuild_features: bool = False,
    promote: bool = False,
) -> RetrainResult:
    loader = DataLoader()
    pipeline = FeaturePipeline()
    trainer = Trainer(pipeline=pipeline)
    evaluator = Evaluator(pipeline=pipeline)
    store = ModelStore()

    raw_df = loader.load_all()
    cache_key = FeaturePipeline.cache_key(
        {
            "purpose": "retrain",
            "rows": len(raw_df),
            "race_min": raw_df["race_id"].min(),
            "race_max": raw_df["race_id"].max(),
            "entry_max": raw_df["race_entry_id"].max(),
            "date_min": raw_df["race_date"].min(),
            "date_max": raw_df["race_date"].max(),
        }
    )
    cache_path = f"data/processed/retrain_features_{cache_key}.parquet"
    if rebuild_features:
        pipeline.invalidate_cache()
    features = pipeline.build_and_cache(raw_df, cache_path)

    splitter = SplitStrategy()
    split = splitter.auto_split(features)
    splits = apply_split(features, split)

    params = trainer.optimize_hyperparams(splits["train"], splits["val"], n_trials=n_trials) if optimize else None
    artifacts = trainer.train_lambdarank(splits["train"], splits["val"], params=params)

    val_scores = _scores_per_race(artifacts.model, splits["val"], artifacts.feature_columns)
    scaler = TemperatureScaler()
    scaler.fit(
        [scores for _, scores, _ in val_scores],
        [winner for _, _, winner in val_scores],
        n_trials=max(10, min(n_trials, 50)),
    )

    candidate_metrics = evaluator.evaluate(artifacts.model, splits["test"], temperature=scaler.temperature)
    importance = evaluator.feature_importance(artifacts.model)
    feature_stats = evaluator.feature_stats(features, artifacts.feature_columns)
    shap_importance = evaluator.shap_analysis(artifacts.model, splits["test"], n_samples=shap_samples)
    metadata = {
        "model_type": artifacts.model_type,
        "params": artifacts.best_params,
        "temperature": scaler.temperature,
        "metrics": candidate_metrics,
        "gate": _gate_result(candidate_metrics),
        "compare_with": compare_with,
        "feature_columns": artifacts.feature_columns,
        "random_seed": trainer.random_seed,
        "library_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "split": {
            "warmup": [split.warmup_start.isoformat(), split.warmup_end.isoformat()],
            "train": [split.train_start.isoformat(), split.train_end.isoformat()],
            "val": [split.val_start.isoformat(), split.val_end.isoformat()],
            "test": [split.test_start.isoformat(), split.test_end.isoformat()],
        },
        "trained_through_date": split.val_end.isoformat(),
        "validation_end_date": split.val_end.isoformat(),
        "data_range": {
            "start": raw_df["race_date"].min().isoformat(),
            "end": raw_df["race_date"].max().isoformat(),
        },
        "feature_cache": str(Path(cache_path).name),
        "candidate": True,
    }
    candidate_version = store.save_candidate(artifacts.model, metadata)
    _write_artifacts(store, candidate_version, importance, feature_stats, shap_importance)

    _, baseline_metadata = store.load(compare_with)
    baseline_version = str(baseline_metadata["version"])
    baseline_metrics = baseline_metadata.get("metrics", {})

    backtest_candidate_roi, backtest_baseline_roi = _compare_recent_backtests(
        store=store,
        compare_window_days=compare_window_days,
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        raw_df=raw_df,
    )

    promoted_version = store.promote(candidate_version) if promote else None
    return RetrainResult(
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        promoted_version=promoted_version,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        backtest_candidate_roi=backtest_candidate_roi,
        backtest_baseline_roi=backtest_baseline_roi,
    )


def _compare_recent_backtests(
    *,
    store: ModelStore,
    compare_window_days: int,
    candidate_version: str,
    baseline_version: str,
    raw_df,
) -> tuple[float | None, float | None]:
    if raw_df.is_empty():
        return None, None
    end_date = raw_df["race_date"].max()
    start_date = end_date - timedelta(days=compare_window_days - 1)
    engine = BacktestEngine(model_store=store)
    candidate_summary = summarize_backtest(
        engine.run(
            start_date=start_date,
            end_date=end_date,
            judgment_clock=datetime.now(UTC).time().replace(hour=10, minute=0, second=0, microsecond=0),
            bankroll=DEFAULT_BANKROLL_JPY,
            evaluation_mode=EvaluationMode.FIXED,
            model_version=candidate_version,
        )
    )
    baseline_summary = summarize_backtest(
        engine.run(
            start_date=start_date,
            end_date=end_date,
            judgment_clock=datetime.now(UTC).time().replace(hour=10, minute=0, second=0, microsecond=0),
            bankroll=DEFAULT_BANKROLL_JPY,
            evaluation_mode=EvaluationMode.FIXED,
            model_version=baseline_version,
        )
    )
    candidate_roi = candidate_summary.roi if candidate_summary.profit_evaluated_races > 0 else None
    baseline_roi = baseline_summary.roi if baseline_summary.profit_evaluated_races > 0 else None
    return candidate_roi, baseline_roi


def _scores_per_race(model, df, feature_columns):
    import pandas as pd
    import polars as pl

    ordered = df.sort(["race_date", "race_number", "race_id", "post_position"])
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
    scores = model.predict(X)
    out = ordered.with_columns(pl.Series("score", scores))
    groups = out.partition_by("race_id", maintain_order=True)
    result = []
    for group in groups:
        winner_idx = None
        for idx, pos in enumerate(group["finish_position"].to_list()):
            if pos == 1:
                winner_idx = idx
                break
        if winner_idx is None:
            continue
        result.append((group["race_id"][0], group["score"].to_numpy(), winner_idx))
    return result


def _gate_result(metrics: dict[str, float]) -> dict[str, object]:
    checks = {
        "win_accuracy": metrics.get("win_accuracy", 0.0) >= 0.25,
        "top3_overlap": metrics.get("top3_overlap", 0.0) >= 0.50,
        "brier_score": metrics.get("brier_score", 1.0) < metrics.get("brier_baseline", 1.0),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_artifacts(store: ModelStore, version: str, importance, feature_stats: dict, shap_importance) -> None:
    version_dir = store.version_dir(version)
    importance.write_csv(version_dir / "feature_importance.csv")
    (version_dir / "feature_stats.json").write_text(_json_dump(feature_stats))
    shap_importance.write_csv(version_dir / "shap_importance.csv")


def _json_dump(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
