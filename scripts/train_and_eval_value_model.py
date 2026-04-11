"""Train Model B (Value) and evaluate the dual-model strategy.

Usage:
    uv run python scripts/train_and_eval_value_model.py --model-a v013
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from datetime import date, time as dt_time
from pathlib import Path

import numpy as np

from providence.backtest.engine import BacktestEngine
from providence.backtest.metrics import summarize_backtest
from providence.domain.enums import TicketType
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.store import ModelStore
from providence.model.value_model import predict_value_scores, train_value_model
from providence.strategy.types import EvaluationMode, StrategyConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="v013")
    args = parser.parse_args()

    store = ModelStore()
    model_a_version = args.model_a
    model_a_dir = store.version_dir(model_a_version)
    model_a_meta = store.load(model_a_version)[1]

    print(f"Model A: {model_a_version}")
    print(f"  win_accuracy: {model_a_meta['metrics']['win_accuracy']:.4f}")

    split = model_a_meta["split"]
    train_end = split["val"][0]
    val_end = split["test"][0]
    test_start = date.fromisoformat(split["test"][0])
    test_end = date.fromisoformat(split["test"][1])

    print(f"\n=== Phase 1: Build features ===")
    loader = DataLoader()
    pipeline = FeaturePipeline()
    raw = loader.load_all()
    print(f"  Raw data: {len(raw)} rows")

    cache_path = Path("data/processed/value_model_features.parquet")
    if cache_path.exists():
        features = __import__("polars").read_parquet(cache_path)
        print(f"  Loaded cached features: {features.shape}")
    else:
        features = pipeline.build_features(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.write_parquet(cache_path)
        print(f"  Built features: {features.shape}")

    print(f"\n=== Phase 2: Train Model B ===")
    model_b, meta_b = train_value_model(
        features,
        train_end=train_end,
        val_end=val_end,
        model_a_path=model_a_version,
    )

    model_b_path = model_a_dir / "value_model_b.txt"
    model_b.save_model(str(model_b_path))
    meta_b_path = model_a_dir / "value_model_b_meta.pkl"
    with open(meta_b_path, "wb") as f:
        pickle.dump(meta_b, f)
    print(f"  Saved: {model_b_path}")

    print(f"\n=== Phase 3: Evaluate dual-model on test set ===")
    import polars as pl

    model_a_booster = store.load(model_a_version)[0]
    if model_a_booster is None:
        models, _, _ = store.load_ensemble(model_a_version)
        from providence.model.ensemble import combine_race_scores
        # For ensemble, we need a wrapper
        class EnsembleWrapper:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            def predict(self, X):
                raw = {k: np.asarray(m.predict(X), dtype=float) for k, m in self.models.items()}
                return combine_race_scores(raw, self.weights)
        from providence.model.ensemble import DEFAULT_WEIGHTS
        model_a_booster = EnsembleWrapper(models, dict(model_a_meta.get("ensemble_weights", DEFAULT_WEIGHTS)))

    model_a_features = model_a_meta["feature_columns"]
    value_features = meta_b["value_features"]

    test_features = features.filter(
        pl.col("race_date").is_between(test_start, test_end, closed="both")
    )
    print(f"  Test races: {test_features['race_id'].n_unique()}, entries: {len(test_features)}")

    race_groups = test_features.partition_by("race_id", maintain_order=True)
    results_rows = []

    for group in race_groups:
        race_id = group["race_id"][0]
        race_date = group["race_date"][0]

        v_scores = predict_value_scores(model_b, group, model_a_booster, model_a_features, value_features)

        finish_positions = group["finish_position"].to_list()
        odds_ranks = group["win_odds_rank"].to_list()
        post_positions = group["post_position"].to_list()

        for i in range(len(group)):
            results_rows.append({
                "race_id": race_id,
                "race_date": race_date,
                "post_position": post_positions[i],
                "finish_position": finish_positions[i],
                "win_odds_rank": odds_ranks[i],
                "value_score_pred": float(v_scores[i]),
            })

    results_df = pl.DataFrame(results_rows)

    # Add model A ranks
    a_scores_all = []
    for group in race_groups:
        from providence.model.value_model import _add_model_a_scores
        scored = _add_model_a_scores(group, model_a_booster, model_a_features)
        a_scores_all.extend(scored["model_a_rank"].to_list())

    results_df = results_df.with_columns(pl.Series("model_a_rank", a_scores_all))

    # Evaluate different α combinations
    print(f"\n=== Phase 4: α sensitivity analysis ===")
    print(f"{'α':>5} {'thresh':>7} {'bets':>5} {'hits':>5} {'hit%':>6} {'profit':>10} {'ROI':>8}")
    print("-" * 55)

    best_profit = -999999
    best_config = None

    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        a_norm = results_df["model_a_rank"].cast(pl.Float64)
        a_score_norm = 1.0 - (a_norm - 1.0) / 7.0  # normalize 1→1.0, 8→0.0

        b_norm = results_df["value_score_pred"].cast(pl.Float64)
        b_max = b_norm.max()
        b_min = b_norm.min()
        b_score_norm = (b_norm - b_min) / (b_max - b_min + 1e-12)

        combined = alpha * a_score_norm + (1 - alpha) * b_score_norm

        for thresh_pct in [0.7, 0.8, 0.85, 0.9]:
            thresh = combined.quantile(thresh_pct)
            if thresh is None:
                continue
            selected = results_df.filter(combined > thresh)
            if selected.is_empty():
                continue

            bets = len(selected)
            hits = selected.filter(pl.col("finish_position") == 1).height
            total_stake = bets * 100
            total_payout = 0.0

            # Simple simulation: bet 100 on each selected runner's win
            # Payout = if finish_position == 1: odds * 100, else 0
            # We don't have odds in results_df, so use a simplified metric
            hit_pct = hits / bets * 100 if bets > 0 else 0

            # Track "value hits": selected runners that beat odds
            value_hits = selected.filter(
                pl.col("finish_position").is_not_null()
                & pl.col("win_odds_rank").is_not_null()
                & (pl.col("finish_position") < pl.col("win_odds_rank"))
            ).height
            value_rate = value_hits / bets * 100 if bets > 0 else 0

            if alpha == 0.5 and thresh_pct == 0.85:
                # Print detail for one representative config
                pass

            if bets >= 10:
                print(f"{alpha:>5.1f} {thresh_pct:>7.0%} {bets:>5} {hits:>5} {hit_pct:>5.1f}% value_beat={value_rate:.0f}%")

    # Compare with Model A alone (backtest)
    print(f"\n=== Phase 5: Backtest comparison (V013 strategy) ===")
    engine = BacktestEngine()
    config = StrategyConfig(
        fractional_kelly=0.05, min_expected_value=0.40, min_confidence=0.90,
        max_candidates=2, max_total_stake=10_000,
        allowed_ticket_types=frozenset([TicketType.WIN]),
    )
    bt_results = engine.run(
        start_date=test_start, end_date=test_end,
        judgment_clock=dt_time(15, 0, 0),
        evaluation_mode=EvaluationMode.FIXED,
        model_version=model_a_version, config=config,
        use_final_odds=False,
    )
    if bt_results:
        s = summarize_backtest(bt_results)
        bet = sum(1 for r in bt_results if r.total_stake > 0)
        hit = sum(1 for r in bt_results if r.total_stake > 0 and any(sr.hit for sr in r.settled_recommendations))
        print(f"  Model A only: profit={s.total_profit:+,.0f}円 ROI={s.roi*100:+.1f}% bet={bet}R hit={hit}")

    print("\nDone.")


if __name__ == "__main__":
    main()
