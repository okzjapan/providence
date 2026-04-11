#!/usr/bin/env python3
"""Train market-independent models (excluding JRDB pre-race indices).

Removes all features that correlate with market sentiment:
- JRDB pre-race indices (IDM, jockey_index, info_index, etc.)
- Features derived from those indices (ranks, relative values)
- JRDB prediction indices (ten/pace/agari/position pred)

Keeps: actual race performance history, horse attributes, race conditions.

Usage:
    uv run python scripts/keiba_train_independent.py --surface turf
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.isotonic import IsotonicRegression

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline
from providence.model.store import ModelStore
from providence.model.trainer import Trainer, _binary_win_labels, _binary_top3_labels

logger = structlog.get_logger()

TRAIN_END = date(2022, 12, 31)
VAL_START = date(2023, 1, 1)
VAL_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 12, 31)

SURFACES = {"turf": 1, "dirt": 2}

MARKET_CORRELATED_FEATURES = {
    # JRDB pre-race indices (directly correlated with market)
    "idm", "jockey_index", "info_index", "training_index",
    "stable_index", "composite_index",
    "upset_index", "longshot_index",
    # JRDB prediction indices (JRDB's algorithmic predictions, priced into market)
    "ten_index_pred", "pace_index_pred", "agari_index_pred", "position_index_pred",
    # Features derived from excluded indices
    "jockey_index_rank", "composite_rank", "training_rank",
    "idm_rank_in_field", "relative_idm", "gap_to_top",
    "field_avg_idm", "field_max_idm", "field_std_idm",
    "ten_pred_rank", "agari_pred_rank", "position_pred_rank",
}

LAMBDARANK_PARAMS = {
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

BINARY_TOP3_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "is_unbalance": True,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbosity": -1,
}


class MarketIndependentPipeline(KeibaFeaturePipeline):
    """Pipeline that excludes market-correlated features."""

    @classmethod
    def feature_columns(cls, df: pl.DataFrame) -> list[str]:
        base = super().feature_columns(df)
        return [c for c in base if c not in MARKET_CORRELATED_FEATURES]


def fit_isotonic(model, val_df, pipeline, label_fn):
    import pandas as pd
    feature_cols = pipeline.feature_columns(val_df)
    cat_cols = pipeline.categorical_columns
    sorted_df = val_df.sort(["race_date", "race_number", "race_id", "post_position"])
    X = pd.DataFrame({
        col: (sorted_df[col].cast(pl.Int32).fill_null(-1).to_list()
              if col in cat_cols
              else sorted_df[col].cast(pl.Float64).fill_null(float("nan")).to_list())
        for col in feature_cols
    })
    raw_preds = model.predict(X)
    y_true = label_fn(sorted_df)
    cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal.fit(raw_preds, y_true)
    return cal


def train_surface(surface_name: str, surface_code: int) -> None:
    log = logger.bind(surface=surface_name)
    log.info("starting_independent_training")

    loader = KeibaDataLoader()
    pipeline = MarketIndependentPipeline()
    trainer = Trainer(pipeline=pipeline)

    log.info("loading_data")
    t0 = time.time()
    all_data = loader.load_race_dataset(end_date=TEST_END)
    surface_data = all_data.filter(all_data["surface_code"] == surface_code)
    log.info("data_loaded", rows=surface_data.shape[0], seconds=f"{time.time()-t0:.1f}")

    log.info("building_features")
    t0 = time.time()
    features = pipeline.build_features(surface_data)

    feature_cols = pipeline.feature_columns(features)
    log.info("features_built",
             rows=features.shape[0], total_cols=features.shape[1],
             model_features=len(feature_cols),
             excluded=len(MARKET_CORRELATED_FEATURES),
             seconds=f"{time.time()-t0:.1f}")
    log.info("feature_list", features=sorted(feature_cols))

    train_df = features.filter(pl.col("race_date") <= TRAIN_END)
    val_df = features.filter((pl.col("race_date") >= VAL_START) & (pl.col("race_date") <= VAL_END))
    test_df = features.filter((pl.col("race_date") >= TEST_START) & (pl.col("race_date") <= TEST_END))
    log.info("split", train=train_df.shape[0], val=val_df.shape[0], test=test_df.shape[0])

    # LambdaRank
    log.info("training_lambdarank_independent")
    t0 = time.time()
    lr = trainer.train_lambdarank(train_df, val_df, params=LAMBDARANK_PARAMS)
    log.info("lambdarank_complete", seconds=f"{time.time()-t0:.1f}")

    # Binary Top3
    log.info("training_binary_top3_independent")
    t0 = time.time()
    bt3 = trainer.train_binary_top3(train_df, val_df, params=BINARY_TOP3_PARAMS)
    log.info("binary_top3_complete", seconds=f"{time.time()-t0:.1f}")

    # Isotonic calibration
    bt3_cal = fit_isotonic(bt3.model, val_df, pipeline, _binary_top3_labels)

    # Save
    store = ModelStore(base_dir=f"data/keiba/models/{surface_name}_independent")
    version_dir = store.base_dir / store._next_version()
    version_dir.mkdir(parents=True, exist_ok=True)
    version = version_dir.name

    lr.model.save_model(str(version_dir / "lambdarank.txt"))
    bt3.model.save_model(str(version_dir / "binary_top3.txt"))
    with open(version_dir / "calibrator_top3.pkl", "wb") as f:
        pickle.dump(bt3_cal, f)

    metadata = {
        "model_type": "market_independent",
        "feature_columns": lr.feature_columns,
        "temperature": 1.0,
        "surface": surface_name,
        "trained_through_date": TRAIN_END.isoformat(),
        "excluded_market_features": sorted(MARKET_CORRELATED_FEATURES),
        "data_range": {"start": "2015-01-01", "end": TEST_END.isoformat()},
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "val_start": VAL_START.isoformat(),
            "val_end": VAL_END.isoformat(),
        },
        "version": version,
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (store.base_dir / "latest").write_text(version)
    log.info("saved", version=version, features=len(lr.feature_columns))

    # Quick divergence check on test set
    import pandas as pd
    test_sorted = test_df.sort(["race_date", "race_number", "race_id", "post_position"])
    X_test = pd.DataFrame({
        col: (test_sorted[col].cast(pl.Int32).fill_null(-1).to_list()
              if col in pipeline.categorical_columns
              else test_sorted[col].cast(pl.Float64).fill_null(float("nan")).to_list())
        for col in lr.feature_columns
    })
    lr_scores = lr.model.predict(X_test)

    odds_list = test_sorted["confirmed_win_odds"].to_list() if "confirmed_win_odds" in test_sorted.columns else [None] * len(lr_scores)
    race_ids = test_sorted["race_id"].to_list()
    finish_list = test_sorted["finish_position"].to_list()

    # Per-race: compare model rank vs market rank
    from collections import defaultdict
    race_data = defaultdict(list)
    for i in range(len(lr_scores)):
        race_data[race_ids[i]].append({
            "score": lr_scores[i],
            "odds": odds_list[i],
            "finish": finish_list[i],
            "idx": i,
        })

    agree_count = 0
    disagree_count = 0
    model_top1_not_fav = 0
    model_top2_has_nonfav = 0
    total_races = 0

    for rid, entries in race_data.items():
        valid = [e for e in entries if e["odds"] and e["odds"] > 0 and e["finish"] is not None]
        if len(valid) < 4:
            continue
        total_races += 1

        by_score = sorted(valid, key=lambda x: -x["score"])
        by_odds = sorted(valid, key=lambda x: x["odds"])

        model_top1 = by_score[0]["idx"]
        market_top1 = by_odds[0]["idx"]

        if model_top1 == market_top1:
            agree_count += 1
        else:
            disagree_count += 1

        model_top1_market_rank = next(i for i, e in enumerate(by_odds) if e["idx"] == model_top1)
        if model_top1_market_rank >= 2:
            model_top1_not_fav += 1

        model_top2_set = {by_score[0]["idx"], by_score[1]["idx"]}
        market_top2_set = {by_odds[0]["idx"], by_odds[1]["idx"]}
        if model_top2_set != market_top2_set:
            model_top2_has_nonfav += 1

    log.info("divergence_analysis",
             total_races=total_races,
             top1_agree=f"{agree_count}/{total_races} ({agree_count/total_races*100:.1f}%)",
             top1_disagree=f"{disagree_count}/{total_races} ({disagree_count/total_races*100:.1f}%)",
             model_top1_not_favorite=f"{model_top1_not_fav}/{total_races} ({model_top1_not_fav/total_races*100:.1f}%)",
             model_top2_differs=f"{model_top2_has_nonfav}/{total_races} ({model_top2_has_nonfav/total_races*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    args = parser.parse_args()

    structlog.configure(processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ])

    surfaces = SURFACES if args.surface == "both" else {args.surface: SURFACES[args.surface]}
    for name, code in surfaces.items():
        train_surface(name, code)


if __name__ == "__main__":
    main()
