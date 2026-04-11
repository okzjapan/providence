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
        self._categorical_columns = FeaturePipeline.categorical_columns

    def train_lambdarank(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="lambdarank", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="lambdarank", reference=train_data, categorical_columns=cat_cols)

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
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_top2", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_top2", reference=train_data, categorical_columns=cat_cols)

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
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_win", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_win", reference=train_data, categorical_columns=cat_cols)

        final_params = {
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
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "binary_win")

    def train_binary_top3(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_top3", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_top3", reference=train_data, categorical_columns=cat_cols)

        final_params = {
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
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "binary_top3")

    def train_focal_value(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
        gamma: float = 2.0,
    ) -> TrainingArtifacts:
        """Train a Focal Loss binary model targeting top-2 finishes.

        Focal Loss down-weights easy-to-classify examples and focuses on
        hard cases (e.g. undervalued runners that finish top-2). This
        produces better-calibrated probabilities in the tails where
        profitable bets live.
        """
        from scipy.special import expit

        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="binary_top2", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="binary_top2", reference=train_data, categorical_columns=cat_cols)

        def focal_objective(preds: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
            y = data.get_label()
            p = expit(preds)
            pt = np.where(y == 1, p, 1.0 - p)
            alpha_t = np.where(y == 1, 0.75, 0.25)
            grad = alpha_t * (
                gamma * (1.0 - pt) ** (gamma - 1.0) * np.log(np.maximum(pt, 1e-12)) * (2.0 * y - 1.0) * p * (1.0 - p)
                + (1.0 - pt) ** gamma * (p - y)
            )
            hess = np.abs(grad) * (1.0 - np.abs(grad))
            hess = np.maximum(hess, 1e-6)
            return grad, hess

        def focal_eval(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
            y = data.get_label()
            p = expit(preds)
            pt = np.where(y == 1, p, 1.0 - p)
            alpha_t = np.where(y == 1, 0.75, 0.25)
            loss = -alpha_t * (1.0 - pt) ** gamma * np.log(np.maximum(pt, 1e-12))
            return "focal_loss", float(np.mean(loss)), False

        final_params = {
            "objective": focal_objective,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbosity": -1,
            "seed": self.random_seed,
            **(params or {}),
        }

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["val"],
            feval=focal_eval,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        saved_params = {k: v for k, v in final_params.items() if k != "objective"}
        saved_params["objective"] = "focal_loss_custom"
        return TrainingArtifacts(model, feature_columns, saved_params, "focal_value")

    def train_xendcg(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        """Train with XE-NDCG-MART objective for smoother probability calibration."""
        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="lambdarank", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="lambdarank", reference=train_data, categorical_columns=cat_cols)

        final_params = {
            "objective": "rank_xendcg",
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
        return TrainingArtifacts(model, feature_columns, final_params, "xendcg")

    def train_huber(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict | None = None,
    ) -> TrainingArtifacts:
        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="huber", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="huber", reference=train_data, categorical_columns=cat_cols)

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
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        return TrainingArtifacts(model, feature_columns, final_params, "huber")

    def optimize_hyperparams(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        n_trials: int = 100,
    ) -> dict[str, float | int]:
        feature_columns = self.pipeline.feature_columns(train_df)
        cat_cols = self._categorical_columns
        train_data = _to_lgb_dataset(train_df, feature_columns, objective="lambdarank", categorical_columns=cat_cols)
        val_data = _to_lgb_dataset(val_df, feature_columns, objective="lambdarank", reference=train_data, categorical_columns=cat_cols)

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
    "binary_top3": "binary_top3",
    "binary_win": "binary_win",
}

_RANKING_OBJECTIVES = {"lambdarank"}


def _to_lgb_dataset(
    df: pl.DataFrame,
    feature_columns: list[str],
    objective: str,
    reference: lgb.Dataset | None = None,
    categorical_columns: list[str] | None = None,
) -> lgb.Dataset:
    if categorical_columns is None:
        categorical_columns = FeaturePipeline.categorical_columns
    sorted_df = df.sort(["race_date", "race_number", "race_id", "post_position"])
    data: dict[str, list] = {}
    for column in feature_columns:
        series = sorted_df[column]
        if column in categorical_columns:
            data[column] = series.cast(pl.Int32).fill_null(-1).to_list()
        else:
            data[column] = series.cast(pl.Float64).fill_null(float("nan")).to_list()
    X = pd.DataFrame(data)

    label_key = _LABEL_FUNCS.get(objective, objective)
    if label_key == "lambdarank":
        y = _lambdarank_labels(sorted_df)
    elif label_key == "binary_top2":
        y = _binary_top2_labels(sorted_df)
    elif label_key == "binary_top3":
        y = _binary_top3_labels(sorted_df)
    elif label_key == "binary_win":
        y = _binary_win_labels(sorted_df)
    else:
        y = _regression_targets(sorted_df)

    group = sorted_df.group_by("race_id").len().sort("race_id")["len"].to_list() if objective in _RANKING_OBJECTIVES else None
    categorical = [col for col in categorical_columns if col in feature_columns]

    weight = None
    if "field_size" in sorted_df.columns:
        weight = (1.0 / sorted_df["field_size"].cast(pl.Float64).fill_null(8.0)).to_list()

    return lgb.Dataset(
        X,
        label=y,
        group=group,
        weight=weight,
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


def _binary_top3_labels(df: pl.DataFrame) -> np.ndarray:
    positions = df["finish_position"].to_list()
    return np.array(
        [1.0 if pos is not None and 1 <= pos <= 3 else 0.0 for pos in positions],
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
