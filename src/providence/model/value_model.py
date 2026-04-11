"""Value Model (Model B): predict market mispricing magnitude.

Learns value_score = max(0, win_odds_rank - finish_position), which
captures HOW MUCH the market underestimated a runner. A high predicted
value_score means the model believes the runner will outperform their
market-implied rank.

Used in combination with Model A (ability ranker) for the dual-model
betting strategy.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from providence.features.pipeline import FeaturePipeline


def train_value_model(
    features: pl.DataFrame,
    *,
    train_end: str,
    val_end: str,
    model_a_path: str | Path,
    random_seed: int = 42,
) -> tuple[lgb.Booster, dict]:
    """Train Model B on value_score target.

    Parameters
    ----------
    features : Full feature DataFrame (must include win_odds_rank, finish_position)
    train_end : ISO date string for train/val split
    val_end : ISO date string for val/test split
    model_a_path : Path to Model A's version directory (for generating ability_vs_market)
    """
    from datetime import date

    from providence.model.store import ModelStore

    train_date = date.fromisoformat(train_end)
    val_date = date.fromisoformat(val_end)

    store = ModelStore()
    model_a_booster, model_a_meta = store.load(str(model_a_path))
    model_a_features = model_a_meta["feature_columns"]

    df = features.filter(
        pl.col("finish_position").is_not_null()
        & pl.col("win_odds_rank").is_not_null()
        & (pl.col("finish_position") >= 1)
        & (pl.col("finish_position") <= 8)
    )

    df = _add_model_a_scores(df, model_a_booster, model_a_features)
    df = _add_value_features(df)

    train_df = df.filter(pl.col("race_date") < train_date)
    val_df = df.filter(
        (pl.col("race_date") >= train_date) & (pl.col("race_date") < val_date)
    )

    value_features = _value_feature_columns(train_df)
    print(f"  Value model features: {len(value_features)}")
    print(f"  Train: {len(train_df)} rows / Val: {len(val_df)} rows")

    cat_cols = [c for c in FeaturePipeline.categorical_columns if c in value_features]
    train_X = _to_pandas(train_df, value_features, cat_cols)
    val_X = _to_pandas(val_df, value_features, cat_cols)

    train_y = _value_score(train_df)
    val_y = _value_score(val_df)

    weight_train = (1.0 / train_df["field_size"].cast(pl.Float64).fill_null(8.0)).to_list()
    weight_val = (1.0 / val_df["field_size"].cast(pl.Float64).fill_null(8.0)).to_list()

    train_data = lgb.Dataset(train_X, label=train_y, weight=weight_train, categorical_feature=cat_cols, free_raw_data=False)
    val_data = lgb.Dataset(val_X, label=val_y, weight=weight_val, categorical_feature=cat_cols, reference=train_data, free_raw_data=False)

    params = {
        "objective": "huber",
        "metric": "huber",
        "huber_delta": 1.0,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "seed": random_seed,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    importance = dict(zip(model.feature_name(), model.feature_importance("gain").tolist()))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    print("  Top 10 features by gain:")
    for name, gain in top_features:
        print(f"    {name}: {gain:.0f}")

    odds_gain = sum(v for k, v in importance.items() if "odds" in k or "market" in k or "ability_vs" in k)
    total_gain = sum(importance.values())
    odds_share = odds_gain / total_gain if total_gain > 0 else 0
    print(f"  Odds-related gain share: {odds_share*100:.1f}%")
    if odds_share > 0.50:
        print("  WARNING: Odds features dominate (>50%). Model may just follow market.")

    metadata = {
        "model_type": "value_model_b",
        "params": params,
        "value_features": value_features,
        "model_a_version": str(model_a_path),
        "odds_gain_share": odds_share,
        "top_features": top_features,
    }

    return model, metadata


def predict_value_scores(
    model_b: lgb.Booster,
    features: pl.DataFrame,
    model_a_booster: lgb.Booster,
    model_a_features: list[str],
    value_features: list[str],
) -> np.ndarray:
    """Predict value_score for each runner."""
    df = _add_model_a_scores(features, model_a_booster, model_a_features)
    df = _add_value_features(df)
    cat_cols = [c for c in FeaturePipeline.categorical_columns if c in value_features]
    X = _to_pandas(df, value_features, cat_cols)
    return model_b.predict(X)


def _add_model_a_scores(df: pl.DataFrame, model_a: lgb.Booster, feature_cols: list[str]) -> pl.DataFrame:
    """Run Model A inference and add scores + rank."""
    import pandas as pd

    cat_cols = FeaturePipeline.categorical_columns
    data = {}
    for col in feature_cols:
        if col in df.columns:
            if col in cat_cols:
                data[col] = df[col].cast(pl.Int32).fill_null(-1).to_list()
            else:
                data[col] = df[col].cast(pl.Float64).fill_null(float("nan")).to_list()
        else:
            data[col] = [float("nan")] * len(df)

    X = pd.DataFrame(data)
    scores = model_a.predict(X)

    out = df.with_columns(pl.Series("model_a_score", scores))
    out = out.with_columns(
        pl.col("model_a_score").rank("dense", descending=True).over("race_id").alias("model_a_rank")
    )
    return out


def _add_value_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add Model B specific features: ability_vs_market, odds_entropy."""
    out = df
    if "model_a_rank" in out.columns and "win_odds_rank" in out.columns:
        out = out.with_columns(
            (pl.col("model_a_rank") - pl.col("win_odds_rank").cast(pl.Float64)).alias("ability_vs_market")
        )

    if "win_odds" in out.columns:
        implied = 1.0 / pl.col("win_odds").cast(pl.Float64).clip(lower_bound=1.01)
        out = out.with_columns(implied.alias("implied_prob"))
        out = out.with_columns(
            (-pl.col("implied_prob") * pl.col("implied_prob").log()).sum().over("race_id").alias("odds_entropy")
        )

    return out


def _value_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature columns for Model B (V013 features + value-specific)."""
    pipeline_features = FeaturePipeline.feature_columns(df)
    value_extras = ["win_odds_rank", "ability_vs_market", "implied_prob", "odds_entropy", "model_a_rank"]
    return [c for c in pipeline_features + value_extras if c in df.columns]


def _value_score(df: pl.DataFrame) -> np.ndarray:
    ranks = df["win_odds_rank"].cast(pl.Float64).to_numpy()
    positions = df["finish_position"].cast(pl.Float64).to_numpy()
    return np.maximum(0, ranks - positions)


def _to_pandas(df: pl.DataFrame, features: list[str], cat_cols: list[str]):
    import pandas as pd

    data = {}
    for col in features:
        if col in df.columns:
            if col in cat_cols:
                data[col] = df[col].cast(pl.Int32).fill_null(-1).to_list()
            else:
                data[col] = df[col].cast(pl.Float64).fill_null(float("nan")).to_list()
        else:
            data[col] = [float("nan")] * len(df)
    return pd.DataFrame(data)
