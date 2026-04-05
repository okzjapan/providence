"""Model training for autorace prediction."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl

from providence.features.pipeline import FeaturePipeline


@dataclass
class TrainingArtifacts:
    model: lgb.Booster
    feature_columns: list[str]
    best_params: dict[str, float | int | str]
    model_type: str


class Trainer:
    """Train LightGBM models for rider strength prediction."""

    def __init__(self, pipeline: FeaturePipeline | None = None, random_seed: int = 42) -> None:
        self.pipeline = pipeline or FeaturePipeline()
        self.random_seed = random_seed

    def train_lambdarank(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="lambdarank")
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="lambdarank", reference=train_data)

        final_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "verbosity": -1,
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "lambdarank")

    def train_binary_top2(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_top2")
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_top2", reference=train_data)

        final_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": True,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "verbosity": -1,
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "binary_top2")

    def train_binary_win(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_win")
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_win", reference=train_data)

        final_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": True,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "verbosity": -1,
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "binary_win")

    def train_huber(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="huber")
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="huber", reference=train_data)

        final_params = {
            "objective": "huber",
            "metric": "huber",
            "huber_delta": 1.0,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "verbosity": -1,
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "huber")

    def optimize_hyperparams(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        n_trials: int = 100,
    ) -> dict[str, float | int]:
        feature_columns = self.pipeline.feature_columns(train_df)
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="lambdarank")
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="lambdarank", reference=train_data)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 3],
                "verbosity": -1,
                "seed": self.random_seed,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": 1,
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )
            score = model.best_score["val"]["ndcg@3"]
            trial.set_user_attr("best_iteration", model.best_iteration)
            return float(score)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_trial.params


_LABEL_FUNCS: dict[str, str] = {
    "lambdarank": "lambdarank",
    "regression": "regression",
    "huber": "regression",
    "binary_top2": "binary_top2",
    "binary_win": "binary_win",
}

_RANKING_OBJECTIVES = {"lambdarank"}


def _to_lgb_dataset(
    df: pl.DataFrame,
    feature_columns: list[str],
    objective: str,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    sorted_df = df.sort(["race_date", "race_number", "race_id", "post_position"])
    data: dict[str, list] = {}
    for column in feature_columns:
        series = sorted_df[column]
        if column in FeaturePipeline.categorical_columns:
            data[column] = series.cast(pl.Int32).fill_null(-1).to_list()
        else:
            data[column] = series.cast(pl.Float64).fill_null(float("nan")).to_list()
    X = pd.DataFrame(data)

    label_key = _LABEL_FUNCS.get(objective, objective)
    if label_key == "lambdarank":
        y = _lambdarank_labels(sorted_df)
    elif label_key == "binary_top2":
        y = _binary_top2_labels(sorted_df)
    elif label_key == "binary_win":
        y = _binary_win_labels(sorted_df)
    else:
        y = _regression_targets(sorted_df)

    group = sorted_df.group_by("race_id").len().sort("race_id")["len"].to_list() if objective in _RANKING_OBJECTIVES else None
    categorical = [col for col in FeaturePipeline.categorical_columns if col in feature_columns]
    return lgb.Dataset(
        X,
        label=y,
        group=group,
        categorical_feature=categorical,
        reference=reference,
        free_raw_data=False,
    )


def _lambdarank_labels(df: pl.DataFrame) -> np.ndarray:
    field_sizes = df["field_size"].to_list()
    positions = df["finish_position"].to_list()
    labels: list[int] = []
    for field_size, position in zip(field_sizes, positions, strict=True):
        if position is None or position in (0, 9):
            labels.append(0)
            continue
        labels.append(max(int(field_size - position), 0))
    return np.array(labels, dtype=int)


def _binary_top2_labels(df: pl.DataFrame) -> np.ndarray:
    positions = df["finish_position"].to_list()
    return np.array(
        [1.0 if pos is not None and 1 <= pos <= 2 else 0.0 for pos in positions],
        dtype=float,
    )


def _binary_win_labels(df: pl.DataFrame) -> np.ndarray:
    positions = df["finish_position"].to_list()
    return np.array(
        [1.0 if pos == 1 else 0.0 for pos in positions],
        dtype=float,
    )


def _regression_targets(df: pl.DataFrame) -> np.ndarray:
    mapping = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
    positions = df["finish_position"].to_list()
    values = [
        mapping.get(position, 0.0) if position is not None else 0.0
        for position in positions
    ]
    return np.array(values, dtype=float)
