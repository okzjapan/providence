#!/usr/bin/env python3
"""Phase 3c: Train 3 models (LambdaRank + BinaryWin + BinaryTop3) with Isotonic calibration.

For each surface (turf/dirt):
  1. Train LambdaRank model (existing approach)
  2. Train Binary Win model (win/not-win)
  3. Train Binary Top3 model (top3/not-top3)
  4. Fit Isotonic Regression calibrators on validation set
  5. Save all models + calibrators

Usage:
    uv run python scripts/keiba_train_phase3c.py
    uv run python scripts/keiba_train_phase3c.py --surface turf
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
from providence.model.trainer import Trainer

logger = structlog.get_logger()

TRAIN_END = date(2022, 12, 31)
VAL_START = date(2023, 1, 1)
VAL_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)
TEST_END = date(2024, 12, 31)

SURFACES = {"turf": 1, "dirt": 2}

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

BINARY_WIN_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "is_unbalance": True,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
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


def fit_isotonic_calibrator(
    model, val_df: pl.DataFrame, pipeline: KeibaFeaturePipeline, label_fn
) -> IsotonicRegression:
    """Fit Isotonic Regression calibrator on validation set predictions."""
    import pandas as pd

    feature_cols = pipeline.feature_columns(val_df)
    cat_cols = pipeline.categorical_columns
    sorted_df = val_df.sort(["race_date", "race_number", "race_id", "post_position"])

    X = pd.DataFrame(
        {
            col: (
                sorted_df[col].cast(pl.Int32).fill_null(-1).to_list()
                if col in cat_cols
                else sorted_df[col].cast(pl.Float64).fill_null(float("nan")).to_list()
            )
            for col in feature_cols
        }
    )

    raw_preds = model.predict(X)
    y_true = label_fn(sorted_df)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_preds, y_true)
    return calibrator


def evaluate_calibration(
    model, test_df: pl.DataFrame, pipeline: KeibaFeaturePipeline, label_fn,
    calibrator: IsotonicRegression | None = None,
) -> dict:
    """Compute calibration metrics on test set."""
    import pandas as pd
    from sklearn.metrics import brier_score_loss, log_loss

    feature_cols = pipeline.feature_columns(test_df)
    cat_cols = pipeline.categorical_columns
    sorted_df = test_df.sort(["race_date", "race_number", "race_id", "post_position"])

    X = pd.DataFrame(
        {
            col: (
                sorted_df[col].cast(pl.Int32).fill_null(-1).to_list()
                if col in cat_cols
                else sorted_df[col].cast(pl.Float64).fill_null(float("nan")).to_list()
            )
            for col in feature_cols
        }
    )

    raw_preds = model.predict(X)
    y_true = label_fn(sorted_df)

    from scipy.special import expit
    raw_probs = expit(raw_preds)

    metrics = {
        "raw_brier": float(brier_score_loss(y_true, raw_probs)),
        "raw_logloss": float(log_loss(y_true, np.clip(raw_probs, 1e-7, 1 - 1e-7))),
    }

    if calibrator is not None:
        cal_probs = calibrator.predict(raw_preds)
        cal_probs = np.clip(cal_probs, 1e-7, 1 - 1e-7)
        metrics["cal_brier"] = float(brier_score_loss(y_true, cal_probs))
        metrics["cal_logloss"] = float(log_loss(y_true, cal_probs))

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(cal_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        cal_table = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_pred = float(cal_probs[mask].mean())
                avg_actual = float(y_true[mask].mean())
                cal_table.append({
                    "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "count": int(mask.sum()),
                    "avg_pred": round(avg_pred, 4),
                    "avg_actual": round(avg_actual, 4),
                })
        metrics["calibration_table"] = cal_table

    return metrics


def train_surface(surface_name: str, surface_code: int) -> None:
    log = logger.bind(surface=surface_name)
    log.info("starting_phase3c_training")

    loader = KeibaDataLoader()
    pipeline = KeibaFeaturePipeline()
    trainer = Trainer(pipeline=pipeline)

    log.info("loading_data")
    t0 = time.time()
    all_data = loader.load_race_dataset(end_date=TEST_END)
    surface_data = all_data.filter(all_data["surface_code"] == surface_code)
    log.info("data_loaded", rows=surface_data.shape[0], seconds=f"{time.time()-t0:.1f}")

    log.info("building_features")
    t0 = time.time()
    features = pipeline.build_features(surface_data)
    log.info("features_built", rows=features.shape[0], cols=features.shape[1], seconds=f"{time.time()-t0:.1f}")

    train_df = features.filter(pl.col("race_date") <= TRAIN_END)
    val_df = features.filter((pl.col("race_date") >= VAL_START) & (pl.col("race_date") <= VAL_END))
    test_df = features.filter((pl.col("race_date") >= TEST_START) & (pl.col("race_date") <= TEST_END))

    log.info("split", train=train_df.shape[0], val=val_df.shape[0], test=test_df.shape[0])

    store = ModelStore(base_dir=f"data/keiba/models/{surface_name}")
    base_metadata = {
        "surface": surface_name,
        "trained_through_date": TRAIN_END.isoformat(),
        "data_range": {"start": "2015-01-01", "end": TEST_END.isoformat()},
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "val_start": VAL_START.isoformat(),
            "val_end": VAL_END.isoformat(),
        },
    }

    # --- 1. LambdaRank ---
    log.info("training_lambdarank")
    t0 = time.time()
    lr_artifacts = trainer.train_lambdarank(train_df, val_df, params=LAMBDARANK_PARAMS)
    log.info("lambdarank_complete", seconds=f"{time.time()-t0:.1f}", features=len(lr_artifacts.feature_columns))

    # --- 2. Binary Win ---
    log.info("training_binary_win")
    t0 = time.time()
    bw_artifacts = trainer.train_binary_win(train_df, val_df, params=BINARY_WIN_PARAMS)
    log.info("binary_win_complete", seconds=f"{time.time()-t0:.1f}")

    # --- 3. Binary Top3 ---
    log.info("training_binary_top3")
    t0 = time.time()
    bt3_artifacts = trainer.train_binary_top3(train_df, val_df, params=BINARY_TOP3_PARAMS)
    log.info("binary_top3_complete", seconds=f"{time.time()-t0:.1f}")

    # --- 4. Isotonic Calibration ---
    log.info("fitting_isotonic_calibrators")

    # --- 4. Huber Regression ---
    log.info("training_huber")
    t0 = time.time()
    huber_artifacts = trainer.train_huber(train_df, val_df)
    log.info("huber_complete", seconds=f"{time.time()-t0:.1f}")

    from providence.model.trainer import _binary_win_labels, _binary_top3_labels

    bw_calibrator = fit_isotonic_calibrator(bw_artifacts.model, val_df, pipeline, _binary_win_labels)
    bt3_calibrator = fit_isotonic_calibrator(bt3_artifacts.model, val_df, pipeline, _binary_top3_labels)
    log.info("calibrators_fitted")

    # --- 5. Evaluate calibration on test set ---
    log.info("evaluating_calibration_on_test")

    bw_cal_metrics = evaluate_calibration(
        bw_artifacts.model, test_df, pipeline, _binary_win_labels, bw_calibrator
    )
    bt3_cal_metrics = evaluate_calibration(
        bt3_artifacts.model, test_df, pipeline, _binary_top3_labels, bt3_calibrator
    )

    log.info(
        "binary_win_calibration",
        raw_brier=f"{bw_cal_metrics['raw_brier']:.4f}",
        cal_brier=f"{bw_cal_metrics.get('cal_brier', 0):.4f}",
        raw_logloss=f"{bw_cal_metrics['raw_logloss']:.4f}",
        cal_logloss=f"{bw_cal_metrics.get('cal_logloss', 0):.4f}",
    )
    log.info(
        "binary_top3_calibration",
        raw_brier=f"{bt3_cal_metrics['raw_brier']:.4f}",
        cal_brier=f"{bt3_cal_metrics.get('cal_brier', 0):.4f}",
        raw_logloss=f"{bt3_cal_metrics['raw_logloss']:.4f}",
        cal_logloss=f"{bt3_cal_metrics.get('cal_logloss', 0):.4f}",
    )

    if "calibration_table" in bw_cal_metrics:
        log.info("binary_win_calibration_table")
        for row in bw_cal_metrics["calibration_table"]:
            log.info("  bin", **row)

    if "calibration_table" in bt3_cal_metrics:
        log.info("binary_top3_calibration_table")
        for row in bt3_cal_metrics["calibration_table"]:
            log.info("  bin", **row)

    # --- 6. Save all models ---
    version_dir = store.base_dir / store._next_version()
    version_dir.mkdir(parents=True, exist_ok=True)
    version = version_dir.name

    lr_artifacts.model.save_model(str(version_dir / "lambdarank.txt"))
    bw_artifacts.model.save_model(str(version_dir / "binary_win.txt"))
    bt3_artifacts.model.save_model(str(version_dir / "binary_top3.txt"))
    huber_artifacts.model.save_model(str(version_dir / "huber.txt"))

    with open(version_dir / "calibrator_win.pkl", "wb") as f:
        pickle.dump(bw_calibrator, f)
    with open(version_dir / "calibrator_top3.pkl", "wb") as f:
        pickle.dump(bt3_calibrator, f)

    ensemble_metadata = {
        **base_metadata,
        "model_type": "phase3c_ensemble",
        "feature_columns": lr_artifacts.feature_columns,
        "temperature": 1.0,
        "models": {
            "lambdarank": {"params": lr_artifacts.best_params, "type": "lambdarank"},
            "binary_win": {"params": bw_artifacts.best_params, "type": "binary_win"},
            "binary_top3": {"params": bt3_artifacts.best_params, "type": "binary_top3"},
            "huber": {"params": huber_artifacts.best_params, "type": "huber"},
        },
        "calibration": {
            "binary_win": {k: v for k, v in bw_cal_metrics.items() if k != "calibration_table"},
            "binary_top3": {k: v for k, v in bt3_cal_metrics.items() if k != "calibration_table"},
        },
        "version": version,
    }

    (version_dir / "metadata.json").write_text(
        json.dumps(ensemble_metadata, indent=2, ensure_ascii=False)
    )
    (store.base_dir / "latest").write_text(version)

    log.info("all_models_saved", version=version, features=len(lr_artifacts.feature_columns))


def main():
    parser = argparse.ArgumentParser(description="Train Phase 3c keiba models")
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
