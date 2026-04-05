"""Phase 3: Odds stacking model.

Trains a simple model that combines base model predictions with market
odds to produce refined win probabilities.  The stacking model learns
WHEN the base model is right vs when the market is right.

Architecture:
  base_model_prob + market_odds features → LogisticRegression → refined_prob

Train: 2025-12 ~ 2026-02 (~1,780 races, ~13k runner observations)
Validate: 2026-03 (~612 races)
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from providence.backtest.settlement import settle_recommendations
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TicketType
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.probability.plackett_luce import compute_win_probs
from providence.strategy.confidence import race_confidence
from providence.strategy.normalize import (
    flatten_ticket_probs,
    market_odds_from_rows,
    payouts_from_rows,
)
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    MarketTicketOdds,
    PredictedTicketProb,
    StrategyConfig,
)

DEV_START = date(2025, 12, 1)
DEV_END = date(2026, 2, 28)
HOLDOUT_START = date(2026, 3, 1)
HOLDOUT_END = date(2026, 3, 31)
JUDGMENT_CLOCK = dt_time(15, 0, 0)

MODEL_SAVE_PATH = Path("data/models/stacking_lr.pkl")


@dataclass
class RunnerRow:
    race_id: int
    race_date: date
    track_id: int
    post_position: int
    model_win_prob: float
    model_rank: int
    market_odds: float
    implied_prob: float
    model_vs_market: float
    confidence: float
    field_size: int
    won: bool


# ── Step 1: Build training data ─────────────────────────────────────

def build_runner_dataset() -> list[RunnerRow]:
    print("=" * 80)
    print("  STEP 1: スタッキング用データセット構築")
    print("=" * 80)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    model_store = ModelStore()
    repo = Repository()
    sf = get_session_factory()

    print("  [1a] 特徴量ロード...", flush=True)
    raw_df = loader.load_race_dataset(end_date=HOLDOUT_END)
    cache_key = FeaturePipeline.cache_key({
        "purpose": "stacking", "rows": len(raw_df),
        "race_min": raw_df["race_id"].min(), "race_max": raw_df["race_id"].max(),
        "entry_max": raw_df["race_entry_id"].max(),
        "date_min": raw_df["race_date"].min(), "date_max": raw_df["race_date"].max(),
        "end_date": HOLDOUT_END.isoformat(),
    })
    cache_path = Path("data/processed") / f"stacking_features_{cache_key}.parquet"
    features = pipeline.build_and_cache(raw_df, str(cache_path))
    dataset = features.filter(
        pl.col("race_date").is_between(DEV_START, HOLDOUT_END, closed="both")
    )

    race_rows = (
        dataset.select(["race_id", "race_date", "track_id", "race_number"])
        .unique(maintain_order=True).sort(["race_date", "race_number", "race_id"])
    )
    unique_dates = sorted(set(r["race_date"] for r in race_rows.iter_rows(named=True)))
    total = race_rows.height

    print(f"       → {len(unique_dates)}日, {total}レース", flush=True)
    print("  [1b] 予測 + オッズ + 結果を結合...", flush=True)

    predictor = Predictor(model_store, pipeline, loader, version="v003")
    all_rows: list[RunnerRow] = []
    done = 0

    for rd in unique_dates:
        day_f = dataset.filter(pl.col("race_date") == rd).sort(
            ["race_number", "race_id", "post_position"]
        )
        bundles = predictor.predict_feature_races(day_f)
        race_ids = list(bundles)
        jt = datetime.combine(rd, JUDGMENT_CLOCK)

        with sf() as session:
            mkt_by = repo.get_latest_market_odds_for_races(session, race_ids, judgment_time=jt)
            pay_by = repo.get_ticket_payouts_for_races(session, race_ids)

        for race_id, bundle in bundles.items():
            done += 1
            mkt_odds_list = market_odds_from_rows(mkt_by.get(race_id, []))
            payouts = payouts_from_rows(pay_by.get(race_id, []))
            if not mkt_odds_list or not payouts:
                continue

            win_payouts = {p.combination: p.payout_value for p in payouts if p.ticket_type == TicketType.WIN}
            win_odds = {o.combination: o.odds_value for o in mkt_odds_list if o.ticket_type == TicketType.WIN}

            scores = bundle.scores
            temperature = bundle.temperature
            win_probs = compute_win_probs(scores, temperature)
            idx_map = bundle.index_map
            n = len(scores)
            confidence = race_confidence(bundle)

            prob_rank = sorted(range(n), key=lambda i: win_probs[i], reverse=True)
            rank_of = {idx: rank + 1 for rank, idx in enumerate(prob_rank)}

            meta = race_rows.filter(pl.col("race_id") == race_id).row(0, named=True)
            track_id = int(meta["track_id"])

            race_df = day_f.filter(pl.col("race_id") == race_id).sort("post_position")
            finish_positions = race_df["finish_position"].to_list()
            post_positions = race_df["post_position"].to_list()

            for i in range(n):
                pp = idx_map.post_position_for_index(i)
                combo = (pp,)
                odds_val = win_odds.get(combo)
                if odds_val is None or odds_val <= 0:
                    continue

                fp_idx = None
                for j, ppos in enumerate(post_positions):
                    if ppos == pp:
                        fp_idx = j
                        break
                won = fp_idx is not None and finish_positions[fp_idx] == 1

                implied = 1.0 / odds_val
                model_p = float(win_probs[i])

                all_rows.append(RunnerRow(
                    race_id=race_id, race_date=rd, track_id=track_id,
                    post_position=pp, model_win_prob=model_p,
                    model_rank=rank_of[i], market_odds=odds_val,
                    implied_prob=implied, model_vs_market=model_p - implied,
                    confidence=confidence, field_size=n, won=won,
                ))

        if done % 200 == 0 or done == total:
            print(f"       {done}/{total} レース", flush=True)

    print(f"  完了: {len(all_rows)} ランナーレコード")
    return all_rows


# ── Step 2: Train stacking model ────────────────────────────────────

def train_stacking(rows: list[RunnerRow]) -> tuple[LogisticRegression, dict]:
    print(f"\n{'=' * 80}")
    print("  STEP 2: スタッキングモデル学習")
    print("=" * 80)

    dev_rows = [r for r in rows if r.race_date <= DEV_END]
    hold_rows = [r for r in rows if r.race_date >= HOLDOUT_START]

    print(f"  開発: {len(dev_rows)} rows | 検証: {len(hold_rows)} rows")

    def to_features(data: list[RunnerRow]) -> np.ndarray:
        return np.array([
            [r.model_win_prob, r.implied_prob, r.model_vs_market,
             r.model_rank, r.confidence, r.field_size, r.market_odds]
            for r in data
        ])

    def to_labels(data: list[RunnerRow]) -> np.ndarray:
        return np.array([1.0 if r.won else 0.0 for r in data])

    X_dev = to_features(dev_rows)
    y_dev = to_labels(dev_rows)
    X_hold = to_features(hold_rows)
    y_hold = to_labels(hold_rows)

    print(f"  特徴量: model_win_prob, implied_prob, model_vs_market, model_rank, confidence, field_size, market_odds")
    print(f"  勝率: 開発 {y_dev.mean()*100:.1f}% | 検証 {y_hold.mean()*100:.1f}%")

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(X_dev, y_dev)

    dev_pred = model.predict_proba(X_dev)[:, 1]
    hold_pred = model.predict_proba(X_hold)[:, 1]

    feature_names = ["model_win_prob", "implied_prob", "model_vs_market",
                     "model_rank", "confidence", "field_size", "market_odds"]

    print(f"\n  係数:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"    {name:<20} {coef:>+8.4f}")
    print(f"    {'intercept':<20} {model.intercept_[0]:>+8.4f}")

    dev_brier = brier_score_loss(y_dev, dev_pred)
    hold_brier = brier_score_loss(y_hold, hold_pred)
    dev_logloss = log_loss(y_dev, dev_pred)
    hold_logloss = log_loss(y_hold, hold_pred)

    base_dev_brier = brier_score_loss(y_dev, np.array([r.model_win_prob for r in dev_rows]))
    base_hold_brier = brier_score_loss(y_hold, np.array([r.model_win_prob for r in hold_rows]))

    print(f"\n  Brier Score (低い方が良い):")
    print(f"    ベースモデル  開発: {base_dev_brier:.5f}  検証: {base_hold_brier:.5f}")
    print(f"    スタッキング  開発: {dev_brier:.5f}  検証: {hold_brier:.5f}")
    print(f"    改善幅        開発: {(base_dev_brier-dev_brier):.5f}  検証: {(base_hold_brier-hold_brier):.5f}")

    metrics = {
        "dev_brier": dev_brier, "hold_brier": hold_brier,
        "base_dev_brier": base_dev_brier, "base_hold_brier": base_hold_brier,
    }
    return model, metrics


# ── Step 3: Evaluate ROI impact ──────────────────────────────────────

def evaluate_roi_impact(
    rows: list[RunnerRow],
    stacking_model: LogisticRegression,
) -> None:
    print(f"\n{'=' * 80}")
    print("  STEP 3: ROI への影響評価")
    print("=" * 80)

    from providence.probability.plackett_luce import compute_all_ticket_probs
    from providence.strategy.bankroll import normalize_to_stakes
    from providence.strategy.candidates import build_candidates
    from providence.strategy.kelly import optimize_kelly_fractions
    from providence.strategy.types import RecommendedBet

    loader = DataLoader()
    pipeline = FeaturePipeline()
    model_store = ModelStore()
    repo = Repository()
    sf = get_session_factory()

    raw_df = loader.load_race_dataset(end_date=HOLDOUT_END)
    cache_key = FeaturePipeline.cache_key({
        "purpose": "stacking", "rows": len(raw_df),
        "race_min": raw_df["race_id"].min(), "race_max": raw_df["race_id"].max(),
        "entry_max": raw_df["race_entry_id"].max(),
        "date_min": raw_df["race_date"].min(), "date_max": raw_df["race_date"].max(),
        "end_date": HOLDOUT_END.isoformat(),
    })
    cache_path = Path("data/processed") / f"stacking_features_{cache_key}.parquet"
    features = pipeline.build_and_cache(raw_df, str(cache_path))

    predictor = Predictor(model_store, pipeline, loader, version="v003")

    best_config = StrategyConfig(
        min_confidence=0.95, min_expected_value=0.3,
        max_candidates=5, fractional_kelly=0.25,
        allowed_ticket_types=frozenset({TicketType.WIN}),
    )

    for period_label, p_start, p_end in [
        ("開発(12-02)", DEV_START, DEV_END),
        ("検証(03)", HOLDOUT_START, HOLDOUT_END),
    ]:
        dataset = features.filter(pl.col("race_date").is_between(p_start, p_end, closed="both"))
        dates = sorted(set(r["race_date"] for r in dataset.select("race_date").unique().iter_rows(named=True)))

        base_stake = 0.0; base_payout = 0.0; base_races = 0; base_hits = 0
        stack_stake = 0.0; stack_payout = 0.0; stack_races = 0; stack_hits = 0

        for rd in dates:
            day_f = dataset.filter(pl.col("race_date") == rd).sort(["race_number", "race_id", "post_position"])
            bundles = predictor.predict_feature_races(day_f)
            jt = datetime.combine(rd, JUDGMENT_CLOCK)

            with sf() as session:
                mkt_by = repo.get_latest_market_odds_for_races(session, list(bundles), judgment_time=jt)
                pay_by = repo.get_ticket_payouts_for_races(session, list(bundles))

            for race_id, bundle in bundles.items():
                mkt_odds_list = market_odds_from_rows(mkt_by.get(race_id, []))
                payouts = payouts_from_rows(pay_by.get(race_id, []))
                if not mkt_odds_list or not payouts:
                    continue

                dc = DecisionContext(judgment_time=jt, evaluation_mode=EvaluationMode.FIXED, provenance="stacking")

                # Base model
                sr_base = run_strategy(bundle, mkt_odds_list, decision_context=dc, config=best_config)
                if sr_base.recommended_bets:
                    settled = settle_recommendations(sr_base.recommended_bets, payouts)
                    base_stake += sum(i.recommendation.recommended_bet for i in settled)
                    base_payout += sum(i.payout_amount for i in settled)
                    base_races += 1
                    if any(i.hit for i in settled):
                        base_hits += 1

                # Stacking: adjust win probabilities, rebuild ticket probs
                confidence = race_confidence(bundle)
                win_odds = {o.combination: o.odds_value for o in mkt_odds_list if o.ticket_type == TicketType.WIN}
                scores = bundle.scores
                n = len(scores)
                win_probs_base = compute_win_probs(scores, bundle.temperature)
                prob_rank = sorted(range(n), key=lambda i: win_probs_base[i], reverse=True)
                rank_of = {idx: rank + 1 for rank, idx in enumerate(prob_rank)}

                adjusted_probs = []
                for i in range(n):
                    pp = bundle.index_map.post_position_for_index(i)
                    odds_val = win_odds.get((pp,))
                    if odds_val and odds_val > 0:
                        implied = 1.0 / odds_val
                        feat = np.array([[
                            float(win_probs_base[i]), implied,
                            float(win_probs_base[i]) - implied,
                            rank_of[i], confidence, n, odds_val
                        ]])
                        adjusted_probs.append(float(stacking_model.predict_proba(feat)[0, 1]))
                    else:
                        adjusted_probs.append(float(win_probs_base[i]))

                # Normalize to sum to 1
                total_adj = sum(adjusted_probs) or 1.0
                adjusted_probs = [p / total_adj for p in adjusted_probs]

                # Create adjusted predicted probs for WIN only
                adj_predicted: list[PredictedTicketProb] = []
                for i in range(n):
                    pp = bundle.index_map.post_position_for_index(i)
                    adj_predicted.append(PredictedTicketProb(
                        ticket_type=TicketType.WIN,
                        combination=(pp,),
                        probability=adjusted_probs[i],
                    ))

                candidates = build_candidates(
                    adj_predicted, mkt_odds_list,
                    confidence_score=confidence, config=best_config,
                )
                if not candidates:
                    continue

                weights = optimize_kelly_fractions(candidates=candidates, bundle=bundle)
                scaled = weights * best_config.fractional_kelly
                recs = [
                    RecommendedBet(
                        ticket_type=c.ticket_type, combination=c.combination,
                        probability=c.probability, odds_value=c.odds_value,
                        expected_value=c.expected_value, confidence_score=c.confidence_score,
                        kelly_fraction=float(w), recommended_bet=0.0, stake_weight=float(w),
                    )
                    for c, w in zip(candidates, scaled) if w > 0
                ]
                rounded = normalize_to_stakes(recs, config=best_config)
                if not rounded:
                    continue

                settled = settle_recommendations(rounded, payouts)
                stack_stake += sum(i.recommendation.recommended_bet for i in settled)
                stack_payout += sum(i.payout_amount for i in settled)
                stack_races += 1
                if any(i.hit for i in settled):
                    stack_hits += 1

        base_roi = base_payout / base_stake * 100 if base_stake > 0 else 0
        stack_roi = stack_payout / stack_stake * 100 if stack_stake > 0 else 0
        base_hr = base_hits / base_races * 100 if base_races > 0 else 0
        stack_hr = stack_hits / stack_races * 100 if stack_races > 0 else 0

        print(f"\n  【{period_label}】 単勝 (conf≥0.95, EV≥0.3)")
        print(f"    ベース:      ROI {base_roi:>6.1f}% 損益 {base_payout-base_stake:>+10,.0f}円 {base_races}R 的中{base_hr:.1f}%")
        print(f"    スタッキング: ROI {stack_roi:>6.1f}% 損益 {stack_payout-stack_stake:>+10,.0f}円 {stack_races}R 的中{stack_hr:.1f}%")


def main() -> None:
    t0 = time.time()

    rows = build_runner_dataset()
    if not rows:
        print("データなし。")
        return

    stacking_model, metrics = train_stacking(rows)

    # Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(stacking_model, f)
    print(f"\n  モデル保存: {MODEL_SAVE_PATH}")

    evaluate_roi_impact(rows, stacking_model)

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")


if __name__ == "__main__":
    main()
