#!/usr/bin/env python3
"""Train LightGBM LambdaRank models for JRA horse racing.

Trains separate models for turf (surface_code=1) and dirt (surface_code=2).

Usage:
    uv run python scripts/keiba_train.py
    uv run python scripts/keiba_train.py --surface turf
    uv run python scripts/keiba_train.py --surface dirt
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import structlog

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline
from providence.model.store import ModelStore
from providence.model.trainer import Trainer

logger = structlog.get_logger()

TRAIN_END = date(2022, 12, 31)
VAL_START = date(2023, 1, 1)
VAL_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 12, 31)

SURFACES = {"turf": 1, "dirt": 2}

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbosity": -1,
}


def train_surface(surface_name: str, surface_code: int) -> None:
    log = logger.bind(surface=surface_name)
    log.info("starting_training")

    loader = KeibaDataLoader()
    pipeline = KeibaFeaturePipeline()

    log.info("loading_data")
    t0 = time.time()
    all_data = loader.load_race_dataset(end_date=TEST_END)
    surface_data = all_data.filter(all_data["surface_code"] == surface_code)
    log.info("data_loaded", rows=surface_data.shape[0], seconds=f"{time.time()-t0:.1f}")

    log.info("building_features")
    t0 = time.time()
    features = pipeline.build_features(surface_data)
    log.info("features_built", rows=features.shape[0], cols=features.shape[1], seconds=f"{time.time()-t0:.1f}")

    import polars as pl

    train_df = features.filter(pl.col("race_date") <= TRAIN_END)
    val_df = features.filter((pl.col("race_date") >= VAL_START) & (pl.col("race_date") <= VAL_END))
    test_df = features.filter((pl.col("race_date") >= TEST_START) & (pl.col("race_date") <= TEST_END))

    log.info("split", train=train_df.shape[0], val=val_df.shape[0], test=test_df.shape[0])

    trainer = Trainer(pipeline=pipeline)
    log.info("training_lambdarank")
    t0 = time.time()
    artifacts = trainer.train_lambdarank(train_df, val_df, params=PARAMS)
    log.info("training_complete", seconds=f"{time.time()-t0:.1f}")

    store = ModelStore(base_dir=f"data/keiba/models/{surface_name}")
    metadata = {
        "feature_columns": artifacts.feature_columns,
        "model_type": artifacts.model_type,
        "best_params": artifacts.best_params,
        "temperature": 1.0,
        "surface": surface_name,
        "trained_through_date": TRAIN_END.isoformat(),
        "data_range": {"start": "2015-01-01", "end": TEST_END.isoformat()},
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "val_start": VAL_START.isoformat(),
            "val_end": VAL_END.isoformat(),
        },
    }
    version = store.save(artifacts.model, metadata)
    log.info("model_saved", version=version, features=len(artifacts.feature_columns))

    log.info("evaluating_on_test_set")
    from providence.model.evaluator import Evaluator

    evaluator = Evaluator(pipeline=pipeline)
    metrics = evaluator.evaluate(artifacts.model, test_df, temperature=1.0)
    for k, v in metrics.items():
        log.info("metric", name=k, value=f"{v:.4f}" if isinstance(v, float) else v)

    fi = evaluator.feature_importance(artifacts.model)
    top_fi = fi.sort("gain", descending=True).head(15)
    log.info("top_features", features=top_fi["feature"].to_list())


def main():
    parser = argparse.ArgumentParser(description="Train keiba LambdaRank models")
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    surfaces = SURFACES if args.surface == "both" else {args.surface: SURFACES[args.surface]}
    for name, code in surfaces.items():
        train_surface(name, code)


if __name__ == "__main__":
    main()
