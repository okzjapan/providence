"""Training CLI command."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import sklearn
import typer
from rich.console import Console

from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.evaluator import Evaluator
from providence.model.split import SplitStrategy, apply_split
from providence.model.store import ModelStore
from providence.model.trainer import Trainer
from providence.probability.calibration import TemperatureScaler

console = Console()


def train_command(
    train_end: str | None = typer.Option(None, help="Manual train end date (YYYY-MM-DD)"),
    val_end: str | None = typer.Option(None, help="Manual validation end date (YYYY-MM-DD)"),
    optimize: bool = typer.Option(False, help="Run Optuna hyperparameter optimization"),
    n_trials: int = typer.Option(30, help="Number of Optuna trials"),
    rebuild_features: bool = typer.Option(False, help="Force rebuild feature cache"),
    compare_with: str | None = typer.Option(None, help="Compare against an existing model version"),
    shap_samples: int = typer.Option(1000, help="Sample size for SHAP importance"),
) -> None:
    if (train_end and not val_end) or (val_end and not train_end):
        console.print("[red]--train-end と --val-end は両方指定するか、両方省略してください。[/red]")
        raise typer.Exit(1)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    trainer = Trainer(pipeline=pipeline)
    evaluator = Evaluator(pipeline=pipeline)
    store = ModelStore()

    raw_df = loader.load_all()
    cache_key = FeaturePipeline.cache_key(
        {
            "rows": len(raw_df),
            "race_min": raw_df["race_id"].min(),
            "race_max": raw_df["race_id"].max(),
            "entry_max": raw_df["race_entry_id"].max(),
            "date_min": raw_df["race_date"].min(),
            "date_max": raw_df["race_date"].max(),
        }
    )
    cache_path = f"data/processed/features_{cache_key}.parquet"
    if rebuild_features:
        pipeline.invalidate_cache()
    features = pipeline.build_and_cache(raw_df, cache_path)

    splitter = SplitStrategy()
    if train_end and val_end:
        split = splitter.manual_split(features, date.fromisoformat(train_end), date.fromisoformat(val_end))
    else:
        split = splitter.auto_split(features)
    splits = apply_split(features, split)

    params = trainer.optimize_hyperparams(splits["train"], splits["val"], n_trials=n_trials) if optimize else None
    artifacts = trainer.train_lambdarank(splits["train"], splits["val"], params=params)

    val_scores = _scores_per_race(artifacts.model, splits["val"], artifacts.feature_columns)
    if not val_scores:
        console.print("[red]validation split から winner を持つレースを抽出できませんでした。[/red]")
        raise typer.Exit(1)
    scaler = TemperatureScaler()
    scaler.fit(
        [scores for _, scores, _ in val_scores],
        [winner for _, _, winner in val_scores],
        n_trials=max(10, min(n_trials, 50)),
    )

    metrics = evaluator.evaluate(artifacts.model, splits["test"], temperature=scaler.temperature)
    importance = evaluator.feature_importance(artifacts.model)
    feature_stats = evaluator.feature_stats(features, artifacts.feature_columns)
    shap_importance = evaluator.shap_analysis(artifacts.model, splits["test"], n_samples=shap_samples)
    gate = _gate_result(metrics)
    compare_result = _compare_with_existing(store, compare_with, metrics) if compare_with else None
    metadata = {
        "model_type": artifacts.model_type,
        "params": artifacts.best_params,
        "temperature": scaler.temperature,
        "metrics": metrics,
        "gate": gate,
        "compare_with": compare_result,
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
        "data_range": {
            "start": raw_df["race_date"].min().isoformat(),
            "end": raw_df["race_date"].max().isoformat(),
        },
        "feature_cache": str(Path(cache_path).name),
    }
    version = store.save(artifacts.model, metadata)
    version_dir = store.version_dir(version)
    importance_path = version_dir / "feature_importance.csv"
    feature_stats_path = version_dir / "feature_stats.json"
    shap_path = version_dir / "shap_importance.csv"
    importance.write_csv(importance_path)
    feature_stats_path.write_text(_json_dump(feature_stats))
    shap_importance.write_csv(shap_path)

    console.print(f"[green]Saved model {version}[/green]")
    console.print(metrics)
    console.print(gate)
    console.print(f"[dim]feature importance saved: {importance_path}[/dim]")
    console.print(f"[dim]feature stats saved: {feature_stats_path}[/dim]")
    console.print(f"[dim]shap importance saved: {shap_path}[/dim]")
    if compare_result:
        console.print(compare_result)


def _scores_per_race(model, df, feature_columns):
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
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def _compare_with_existing(store: ModelStore, version: str, new_metrics: dict[str, float]) -> dict[str, object]:
    try:
        _, metadata = store.load(version)
    except FileNotFoundError:
        return {"version": version, "error": "not_found"}

    old_metrics = metadata.get("metrics", {})
    comparison = {}
    for key in ("win_accuracy", "top3_overlap", "brier_score"):
        if key in old_metrics and key in new_metrics:
            comparison[key] = {
                "old": old_metrics[key],
                "new": new_metrics[key],
                "delta": new_metrics[key] - old_metrics[key],
            }
    return {"version": version, "comparison": comparison}


def _json_dump(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
