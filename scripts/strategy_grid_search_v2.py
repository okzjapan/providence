"""Strategy parameter grid search v2.

券種別にROI > 0%を達成する最適な戦略パラメータを探索する。
オッズソースは odds_snapshot（全組み合わせ）を使用。
ticket_payouts は精算（的中判定）にのみ使用。

Usage:
    uv run python scripts/strategy_grid_search_v2.py --model-version v008
    uv run python scripts/strategy_grid_search_v2.py --model-version v010
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path

import numpy as np
import polars as pl

from providence.backtest.settlement import settle_recommendations
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TicketType, TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.strategy.bankroll import normalize_to_stakes
from providence.strategy.candidates import build_candidates
from providence.strategy.confidence import race_confidence
from providence.strategy.kelly import optimize_kelly_fractions
from providence.strategy.normalize import (
    flatten_ticket_probs,
    market_odds_from_rows,
    payouts_from_rows,
)
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    MarketTicketOdds,
    RacePredictionBundle,
    RecommendedBet,
    SettledTicketPayout,
    StrategyConfig,
    TicketCandidate,
)

DEV_START = date(2025, 5, 1)
DEV_END = date(2025, 12, 31)
HOLDOUT_START = date(2026, 1, 1)
HOLDOUT_END = date(2026, 4, 30)
JUDGMENT_CLOCK = dt_time(15, 0, 0)

TICKET_TYPES = [
    TicketType.WIN, TicketType.EXACTA, TicketType.QUINELLA,
    TicketType.WIDE, TicketType.TRIFECTA, TicketType.TRIO,
]
TT_LABELS = {tt: tt.value for tt in TICKET_TYPES}

CONFIDENCE_VALS = [0.0, 0.80, 0.90, 0.95, 0.97, 0.99]
EV_VALS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
CAND_VALS = [2, 3, 5, 8]
KELLY_VALS = [0.05, 0.10, 0.25]

MIN_BET_RACES = 15

MIN_HIT_RATE_BY_TICKET: dict[TicketType, float] = {
    TicketType.WIN: 0.125,
    TicketType.EXACTA: 0.0179,
    TicketType.QUINELLA: 0.0357,
    TicketType.WIDE: 0.1071,
    TicketType.TRIFECTA: 0.003,
    TicketType.TRIO: 0.0179,
}


@dataclass
class RaceCache:
    race_id: int
    bundle: RacePredictionBundle
    market_odds: list[MarketTicketOdds]
    payouts: list[SettledTicketPayout]
    judgment_time: datetime
    confidence: float
    track_id: int
    race_date: date


@dataclass
class EvalResult:
    ticket_type: TicketType
    min_confidence: float
    min_ev: float
    max_candidates: int
    fractional_kelly: float
    races_bet: int
    total_bets: int
    total_stake: float
    total_payout: float
    hit_races: int

    @property
    def roi(self) -> float:
        return (self.total_payout - self.total_stake) / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.total_payout - self.total_stake

    @property
    def hit_rate(self) -> float:
        return self.hit_races / self.races_bet if self.races_bet > 0 else 0.0


# ── Precompute ───────────────────────────────────────────────────────

def precompute(model_version: str) -> dict[int, RaceCache]:
    print("=" * 80)
    print(f"  STEP 1: 事前計算 (model={model_version})")
    print("=" * 80)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    model_store = ModelStore()
    repository = Repository()
    session_factory = get_session_factory()

    print("  [1a] 特徴量ロード...", flush=True)
    raw_df = loader.load_race_dataset(end_date=HOLDOUT_END)
    if raw_df.is_empty():
        return {}
    cache_key = FeaturePipeline.cache_key({
        "purpose": "strategy_grid_v2", "rows": len(raw_df),
        "race_min": raw_df["race_id"].min(), "race_max": raw_df["race_id"].max(),
        "entry_max": raw_df["race_entry_id"].max(),
        "end_date": HOLDOUT_END.isoformat(),
    })
    cache_path = Path("data/processed") / f"strategy_v2_{cache_key}.parquet"
    features = pipeline.build_and_cache(raw_df, str(cache_path))
    dataset = features.filter(pl.col("race_date").is_between(DEV_START, HOLDOUT_END, closed="both"))
    if dataset.is_empty():
        return {}

    race_rows = (
        dataset.select(["race_id", "race_date", "track_id", "race_number"])
        .unique(maintain_order=True).sort(["race_date", "race_number", "race_id"])
    )
    unique_dates = sorted({r["race_date"] for r in race_rows.iter_rows(named=True)})
    total = race_rows.height
    print(f"       → {len(unique_dates)}日, {total}レース", flush=True)

    print(f"  [1b] モデル予測 + オッズ/払戻取得 (model={model_version})...", flush=True)
    predictor = Predictor(model_store, pipeline, loader, version=model_version)
    caches: dict[int, RaceCache] = {}
    done = 0
    skip = 0
    t0 = time.time()

    for rd in unique_dates:
        day_f = dataset.filter(pl.col("race_date") == rd).sort(["race_number", "race_id", "post_position"])
        bundles = predictor.predict_feature_races(day_f)
        race_ids = list(bundles)
        jt = datetime.combine(rd, JUDGMENT_CLOCK)

        with session_factory() as session:
            mkt_by = repository.get_latest_market_odds_for_races(session, race_ids, judgment_time=jt)
            pay_by = repository.get_ticket_payouts_for_races(session, race_ids)

        for race_id, bundle in bundles.items():
            done += 1
            odds = market_odds_from_rows(mkt_by.get(race_id, []))
            payouts = payouts_from_rows(pay_by.get(race_id, []))
            if not odds or not payouts:
                skip += 1
                continue
            meta = race_rows.filter(pl.col("race_id") == race_id).row(0, named=True)
            caches[race_id] = RaceCache(
                race_id=race_id, bundle=bundle, market_odds=odds, payouts=payouts,
                judgment_time=jt, confidence=race_confidence(bundle),
                track_id=int(meta["track_id"]), race_date=rd,
            )

        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"       {done}/{total} ({done/total*100:.0f}%) skip:{skip} {elapsed:.0f}s ETA:{eta:.0f}s", flush=True)

    print(f"  完了: {len(caches)} 有効レース ({time.time()-t0:.0f}s)")
    return caches


# ── Grid Search ──────────────────────────────────────────────────────

def kelly_for_race(rc: RaceCache, tt: TicketType, min_ev: float, max_cand: int):
    config = StrategyConfig(min_expected_value=min_ev, max_candidates=max_cand, allowed_ticket_types=frozenset({tt}))
    predicted = flatten_ticket_probs(rc.bundle.ticket_probs, rc.bundle.index_map)
    candidates = build_candidates(predicted, rc.market_odds, confidence_score=rc.confidence, config=config)
    if not candidates:
        return None
    weights = optimize_kelly_fractions(candidates=candidates, bundle=rc.bundle)
    if np.all(weights <= 0):
        return None
    return candidates, weights


def settle_with_kelly(candidates, raw_weights, kf, payouts, config):
    scaled = raw_weights * kf
    recs = [
        RecommendedBet(
            ticket_type=c.ticket_type, combination=c.combination,
            probability=c.probability, odds_value=c.odds_value,
            expected_value=c.expected_value, confidence_score=c.confidence_score,
            kelly_fraction=float(w), recommended_bet=0.0, stake_weight=float(w),
        )
        for c, w in zip(candidates, scaled) if w > 0
    ]
    rounded = normalize_to_stakes(recs, config=config)
    if not rounded:
        return 0.0, 0.0, 0, False
    settled = settle_recommendations(rounded, payouts)
    stake = sum(s.recommendation.recommended_bet for s in settled)
    payout = sum(s.payout_amount for s in settled)
    any_hit = any(s.hit for s in settled)
    return stake, payout, len(settled), any_hit


def grid_search(caches: dict[int, RaceCache], date_start: date, date_end: date, label: str) -> list[EvalResult]:
    print(f"\n{'=' * 80}")
    print(f"  STEP 2: グリッドサーチ [{label}] ({date_start}〜{date_end})")
    print("=" * 80)

    races = {k: v for k, v in caches.items() if date_start <= v.race_date <= date_end}
    combos = list(itertools.product(CONFIDENCE_VALS, EV_VALS, CAND_VALS))
    total_evals = len(TICKET_TYPES) * len(combos) * len(KELLY_VALS)
    print(f"  レース数: {len(races)} | 評価数: {total_evals}")

    all_results: list[EvalResult] = []
    eval_done = 0
    t0 = time.time()

    for tt in TICKET_TYPES:
        tt_best_profit = float("-inf")
        tt_best_label = ""

        for conf, ev, mc in combos:
            race_kelly_cache = []
            for rc in races.values():
                if rc.confidence < conf:
                    continue
                result = kelly_for_race(rc, tt, ev, mc)
                if result is not None:
                    race_kelly_cache.append((rc, result[0], result[1]))

            for kf in KELLY_VALS:
                eval_done += 1
                stake_config = StrategyConfig(fractional_kelly=kf, max_total_stake=10_000)
                total_stake = 0.0
                total_payout = 0.0
                total_bets = 0
                races_bet = 0
                hit_races = 0

                for rc, cands, weights in race_kelly_cache:
                    s, p, nb, hit = settle_with_kelly(cands, weights, kf, rc.payouts, stake_config)
                    if nb > 0:
                        total_stake += s
                        total_payout += p
                        total_bets += nb
                        races_bet += 1
                        if hit:
                            hit_races += 1

                er = EvalResult(
                    ticket_type=tt, min_confidence=conf, min_ev=ev, max_candidates=mc,
                    fractional_kelly=kf, races_bet=races_bet, total_bets=total_bets,
                    total_stake=total_stake, total_payout=total_payout, hit_races=hit_races,
                )
                all_results.append(er)

                if er.races_bet >= MIN_BET_RACES and er.profit > tt_best_profit:
                    tt_best_profit = er.profit
                    tt_best_label = f"conf≥{conf} EV≥{ev} cand≤{mc} kelly={kf} ROI={er.roi*100:+.1f}%"

            if eval_done % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / eval_done * (total_evals - eval_done) if eval_done > 0 else 0
                print(f"    [{eval_done}/{total_evals}] {elapsed:.0f}s ETA:{eta:.0f}s | {TT_LABELS[tt]} best: {tt_best_label}", flush=True)

        print(f"  {TT_LABELS[tt]:>6}: profit={tt_best_profit:+,.0f}円  {tt_best_label}", flush=True)

    print(f"  グリッドサーチ完了 ({time.time()-t0:.0f}s)")
    return all_results


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(results: list[EvalResult], label: str) -> dict[TicketType, list[EvalResult]]:
    print(f"\n{'=' * 80}")
    print(f"  STEP 3: 結果分析 [{label}]")
    print("=" * 80)

    best_by_tt: dict[TicketType, list[EvalResult]] = {}

    for tt in TICKET_TYPES:
        min_hr = MIN_HIT_RATE_BY_TICKET.get(tt, 0.0)
        valid = [
            r for r in results
            if r.ticket_type == tt and r.races_bet >= MIN_BET_RACES and r.hit_rate >= min_hr and r.roi > 0
        ]
        valid.sort(key=lambda r: r.profit, reverse=True)
        best_by_tt[tt] = valid[:5]

        print(f"\n  【{TT_LABELS[tt]}】 ROI>0% かつ profit上位5:")
        if not valid:
            print(f"    → 該当なし")
            continue
        print(f"    {'#':>3} {'ROI':>8} {'profit':>12} {'投下額':>10} {'betR':>6} {'hitR':>6} {'hit%':>6}  パラメータ")
        for i, r in enumerate(valid[:5], 1):
            print(
                f"    {i:>3} {r.roi*100:>+7.1f}% {r.profit:>+11,.0f}円 {r.total_stake:>9,.0f}円 "
                f"{r.races_bet:>6} {r.hit_races:>6} {r.hit_rate*100:>5.1f}%  "
                f"conf≥{r.min_confidence} EV≥{r.min_ev} cand≤{r.max_candidates} kelly={r.fractional_kelly}"
            )

    return best_by_tt


def holdout_validation(caches: dict[int, RaceCache], dev_best: dict[TicketType, list[EvalResult]]) -> dict:
    print(f"\n{'=' * 80}")
    print(f"  STEP 4: ホールドアウト検証 ({HOLDOUT_START}〜{HOLDOUT_END})")
    print("=" * 80)

    hold = {k: v for k, v in caches.items() if v.race_date >= HOLDOUT_START}
    print(f"  検証レース数: {len(hold)}")

    report: dict[str, dict] = {}

    for tt in TICKET_TYPES:
        candidates = dev_best.get(tt, [])
        if not candidates:
            print(f"  {TT_LABELS[tt]:>6}: 開発セットで ROI>0% なし → スキップ")
            report[TT_LABELS[tt]] = {"status": "no_positive_roi_in_dev"}
            continue

        for rank, dev_er in enumerate(candidates[:3], 1):
            er = _eval_on_races(hold, tt, dev_er.min_confidence, dev_er.min_ev, dev_er.max_candidates, dev_er.fractional_kelly)
            mark = " ★" if er.races_bet >= 3 and er.roi > 0 else ""
            print(
                f"  {TT_LABELS[tt]:>6} #{rank}: 開発ROI={dev_er.roi*100:>+.1f}% profit={dev_er.profit:>+,.0f}円 "
                f"→ 検証ROI={er.roi*100:>+.1f}% profit={er.profit:>+,.0f}円 betR={er.races_bet}{mark}"
            )
            if rank == 1:
                report[TT_LABELS[tt]] = {
                    "status": "validated" if er.roi > 0 and er.races_bet >= 3 else "not_validated",
                    "dev": {"roi": dev_er.roi, "profit": dev_er.profit, "params": {
                        "min_confidence": dev_er.min_confidence, "min_ev": dev_er.min_ev,
                        "max_candidates": dev_er.max_candidates, "fractional_kelly": dev_er.fractional_kelly,
                    }},
                    "holdout": {"roi": er.roi, "profit": er.profit, "races_bet": er.races_bet},
                }

    return report


def _eval_on_races(races, tt, conf, ev, mc, kf) -> EvalResult:
    stake_config = StrategyConfig(fractional_kelly=kf, max_total_stake=10_000)
    total_stake = 0.0
    total_payout = 0.0
    total_bets = 0
    races_bet = 0
    hit_races = 0
    for rc in races.values():
        if rc.confidence < conf:
            continue
        result = kelly_for_race(rc, tt, ev, mc)
        if result is None:
            continue
        cands, weights = result
        s, p, nb, hit = settle_with_kelly(cands, weights, kf, rc.payouts, stake_config)
        if nb > 0:
            total_stake += s
            total_payout += p
            total_bets += nb
            races_bet += 1
            if hit:
                hit_races += 1
    return EvalResult(
        ticket_type=tt, min_confidence=conf, min_ev=ev, max_candidates=mc,
        fractional_kelly=kf, races_bet=races_bet, total_bets=total_bets,
        total_stake=total_stake, total_payout=total_payout, hit_races=hit_races,
    )


# ── Save ─────────────────────────────────────────────────────────────

def save_results(model_version: str, dev_best: dict, holdout_report: dict, all_results: list[EvalResult]):
    out_dir = Path("data/strategy_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{timestamp}_{model_version}"

    json_data = {
        "model_version": model_version,
        "dev_period": [DEV_START.isoformat(), DEV_END.isoformat()],
        "holdout_period": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "search_params": {
            "confidence": CONFIDENCE_VALS, "ev": EV_VALS,
            "candidates": CAND_VALS, "kelly": KELLY_VALS,
        },
        "holdout_report": holdout_report,
        "dev_best": {
            TT_LABELS[tt]: [
                {"roi": r.roi, "profit": r.profit, "stake": r.total_stake,
                 "races_bet": r.races_bet, "hit_rate": r.hit_rate,
                 "params": {"conf": r.min_confidence, "ev": r.min_ev, "cand": r.max_candidates, "kelly": r.fractional_kelly}}
                for r in ers
            ]
            for tt, ers in dev_best.items()
        },
    }
    json_path = out_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    print(f"\n  結果保存: {json_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True, help="Model version (e.g. v008, v010)")
    args = parser.parse_args()

    t_start = time.time()
    print(f"\n  モデル: {args.model_version}")
    print(f"  開発: {DEV_START}〜{DEV_END} / 検証: {HOLDOUT_START}〜{HOLDOUT_END}")
    print(f"  探索: {len(TICKET_TYPES)}券種 x {len(CONFIDENCE_VALS)}conf x {len(EV_VALS)}EV x {len(CAND_VALS)}cand x {len(KELLY_VALS)}kelly = {len(TICKET_TYPES)*len(CONFIDENCE_VALS)*len(EV_VALS)*len(CAND_VALS)*len(KELLY_VALS)}通り")

    caches = precompute(args.model_version)
    if not caches:
        print("データなし。")
        sys.exit(1)

    dev_count = sum(1 for v in caches.values() if v.race_date <= DEV_END)
    hold_count = sum(1 for v in caches.values() if v.race_date >= HOLDOUT_START)
    print(f"\n  開発: {dev_count}R | ホールドアウト: {hold_count}R")

    dev_results = grid_search(caches, DEV_START, DEV_END, "開発セット")
    dev_best = analyze(dev_results, "開発セット")
    holdout_report = holdout_validation(caches, dev_best)
    save_results(args.model_version, dev_best, holdout_report, dev_results)

    total = time.time() - t_start
    print(f"\n総実行時間: {total:.0f}秒 ({total/60:.1f}分)")


if __name__ == "__main__":
    main()
