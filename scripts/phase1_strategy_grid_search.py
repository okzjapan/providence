"""Phase 1: Strategy parameter grid search with full Kelly optimization.

Architecture:
  1. Pre-compute predictions + market data once (~2 min)
  2. For each (ticket_type, confidence, ev_threshold, max_candidates):
     - Run Kelly optimization once per race (~40 min total)
     - Try 3 fractional_kelly values via re-scaling (cheap)
  3. Track analysis on top configs (~5 min)
  4. Holdout validation (~5 min)

Total estimated: ~50 min. Progress shown at every step.
"""

from __future__ import annotations

import itertools
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

# ── Constants ────────────────────────────────────────────────────────

DEV_END = date(2026, 2, 28)
HOLDOUT_START = date(2026, 3, 1)
HOLDOUT_END = date(2026, 3, 31)
FULL_START = date(2025, 8, 8)
JUDGMENT_CLOCK = dt_time(15, 0, 0)

TICKET_TYPES = [
    TicketType.WIN, TicketType.EXACTA, TicketType.QUINELLA,
    TicketType.WIDE, TicketType.TRIFECTA, TicketType.TRIO,
]
TT_LABELS = {tt: tt.value for tt in TICKET_TYPES}
TRACK_NAMES = {tc.value: tc.japanese_name for tc in TrackCode}

CONFIDENCE_VALS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97]
EV_VALS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
CAND_VALS = [2, 3, 5, 8, 12]
KELLY_VALS = [0.10, 0.25, 0.50]


# ── Data types ───────────────────────────────────────────────────────

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
    track_ids: list[int] | None
    races_bet: int
    total_bets: int
    total_stake: float
    total_payout: float
    hit_races: int

    @property
    def roi(self) -> float:
        return self.total_payout / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.total_payout - self.total_stake

    @property
    def hit_rate(self) -> float:
        return self.hit_races / self.races_bet if self.races_bet > 0 else 0.0


# ── Step 1: Pre-compute ─────────────────────────────────────────────

def precompute(end_date: date) -> dict[int, RaceCache]:
    print("=" * 90)
    print("  STEP 1: 予測 + 市場データの事前計算")
    print("=" * 90)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    model_store = ModelStore()
    repository = Repository()
    session_factory = get_session_factory()

    print("  [1a] 特徴量ロード...", flush=True)
    raw_df = loader.load_race_dataset(end_date=end_date)
    if raw_df.is_empty():
        return {}
    cache_key = FeaturePipeline.cache_key({
        "purpose": "backtest", "rows": len(raw_df),
        "race_min": raw_df["race_id"].min(), "race_max": raw_df["race_id"].max(),
        "entry_max": raw_df["race_entry_id"].max(),
        "date_min": raw_df["race_date"].min(), "date_max": raw_df["race_date"].max(),
        "end_date": end_date.isoformat(),
    })
    cache_path = Path("data/processed") / f"backtest_features_{cache_key}.parquet"
    features = pipeline.build_and_cache(raw_df, str(cache_path))
    dataset = features.filter(
        pl.col("race_date").is_between(FULL_START, end_date, closed="both")
    )
    if dataset.is_empty():
        return {}

    race_rows = (
        dataset.select(["race_id", "race_date", "track_id", "race_number"])
        .unique(maintain_order=True)
        .sort(["race_date", "race_number", "race_id"])
    )
    unique_dates = sorted(set(
        r["race_date"] for r in race_rows.iter_rows(named=True)
    ))
    total = race_rows.height
    print(f"       → {len(unique_dates)}日, {total}レース", flush=True)

    print("  [1b] モデル予測 + オッズ/払戻取得...", flush=True)
    predictor = Predictor(model_store, pipeline, loader, version="latest")
    caches: dict[int, RaceCache] = {}
    done = 0
    skip = 0
    t0 = time.time()

    for rd in unique_dates:
        day_features = dataset.filter(pl.col("race_date") == rd).sort(
            ["race_number", "race_id", "post_position"]
        )
        bundles = predictor.predict_feature_races(day_features)
        race_ids = list(bundles)
        jt = datetime.combine(rd, JUDGMENT_CLOCK)

        with session_factory() as session:
            mkt_by_race = repository.get_latest_market_odds_for_races(
                session, race_ids, judgment_time=jt,
            )
            pay_by_race = repository.get_ticket_payouts_for_races(session, race_ids)

        for race_id, bundle in bundles.items():
            done += 1
            odds = market_odds_from_rows(mkt_by_race.get(race_id, []))
            payouts = payouts_from_rows(pay_by_race.get(race_id, []))
            if not odds or not payouts:
                skip += 1
                continue
            meta = race_rows.filter(pl.col("race_id") == race_id).row(0, named=True)
            caches[race_id] = RaceCache(
                race_id=race_id,
                bundle=bundle,
                market_odds=odds,
                payouts=payouts,
                judgment_time=jt,
                confidence=race_confidence(bundle),
                track_id=int(meta["track_id"]),
                race_date=rd,
            )

        elapsed = time.time() - t0
        pct = done / total * 100
        eta = elapsed / done * (total - done) if done > 0 else 0
        print(f"       {done}/{total} ({pct:.0f}%) スキップ:{skip} 経過:{elapsed:.0f}s 残り:{eta:.0f}s", flush=True)

    print(f"  完了: {len(caches)} 有効レース ({time.time()-t0:.0f}s)")
    return caches


# ── Step 2: Grid search ─────────────────────────────────────────────

def kelly_for_race(
    rc: RaceCache,
    ticket_type: TicketType,
    min_ev: float,
    max_candidates: int,
) -> tuple[list[TicketCandidate], np.ndarray] | None:
    """Run Kelly optimization for one race. Returns (candidates, raw_weights)."""
    config = StrategyConfig(
        min_expected_value=min_ev,
        max_candidates=max_candidates,
        allowed_ticket_types=frozenset({ticket_type}),
    )
    predicted = flatten_ticket_probs(rc.bundle.ticket_probs, rc.bundle.index_map)
    candidates = build_candidates(
        predicted, rc.market_odds,
        confidence_score=rc.confidence, config=config,
    )
    if not candidates:
        return None
    weights = optimize_kelly_fractions(candidates=candidates, bundle=rc.bundle)
    if np.all(weights <= 0):
        return None
    return candidates, weights


def settle_with_kelly(
    candidates: list[TicketCandidate],
    raw_weights: np.ndarray,
    fractional_kelly: float,
    payouts: list[SettledTicketPayout],
    config: StrategyConfig,
) -> tuple[float, float, int, bool]:
    """Apply kelly fraction, normalize stakes, settle. Returns (stake, payout, n_bets, any_hit)."""
    scaled = raw_weights * fractional_kelly
    recs = [
        RecommendedBet(
            ticket_type=c.ticket_type, combination=c.combination,
            probability=c.probability, odds_value=c.odds_value,
            expected_value=c.expected_value, confidence_score=c.confidence_score,
            kelly_fraction=float(w), recommended_bet=0.0, stake_weight=float(w),
        )
        for c, w in zip(candidates, scaled)
        if w > 0
    ]
    rounded = normalize_to_stakes(recs, config=config)
    if not rounded:
        return 0.0, 0.0, 0, False
    settled = settle_recommendations(rounded, payouts)
    stake = sum(s.recommendation.recommended_bet for s in settled)
    payout = sum(s.payout_amount for s in settled)
    n_bets = len(settled)
    any_hit = any(s.hit for s in settled)
    return stake, payout, n_bets, any_hit


def grid_search(
    caches: dict[int, RaceCache],
    date_start: date,
    date_end: date,
    label: str,
) -> list[EvalResult]:
    print(f"\n{'=' * 90}")
    print(f"  STEP 2: グリッドサーチ [{label}] ({date_start}〜{date_end})")
    print("=" * 90)

    races = {k: v for k, v in caches.items() if date_start <= v.race_date <= date_end}
    n_races = len(races)

    combos = list(itertools.product(CONFIDENCE_VALS, EV_VALS, CAND_VALS))
    n_kelly = len(combos)
    n_total = len(TICKET_TYPES) * n_kelly * len(KELLY_VALS)
    print(f"  レース数: {n_races}")
    print(f"  パラメータ: {len(TICKET_TYPES)}券種 × {n_kelly} Kelly評価 × {len(KELLY_VALS)} kelly率 = {n_total} 評価")

    all_results: list[EvalResult] = []
    kelly_done = 0
    kelly_total = len(TICKET_TYPES) * n_kelly
    t0 = time.time()

    for tt in TICKET_TYPES:
        tt_best_roi = 0.0
        tt_best_label = ""

        for conf, ev, mc in combos:
            kelly_done += 1

            race_kelly_cache: list[tuple[RaceCache, list[TicketCandidate], np.ndarray]] = []
            for rc in races.values():
                if rc.confidence < conf:
                    continue
                result = kelly_for_race(rc, tt, ev, mc)
                if result is not None:
                    race_kelly_cache.append((rc, result[0], result[1]))

            for kf in KELLY_VALS:
                stake_config = StrategyConfig(
                    fractional_kelly=kf,
                    max_total_stake=10_000,
                )
                total_stake = 0.0
                total_payout = 0.0
                total_bets = 0
                races_bet = 0
                hit_races = 0

                for rc, cands, weights in race_kelly_cache:
                    s, p, nb, hit = settle_with_kelly(
                        cands, weights, kf, rc.payouts, stake_config,
                    )
                    if nb > 0:
                        total_stake += s
                        total_payout += p
                        total_bets += nb
                        races_bet += 1
                        if hit:
                            hit_races += 1

                er = EvalResult(
                    ticket_type=tt, min_confidence=conf, min_ev=ev,
                    max_candidates=mc, fractional_kelly=kf, track_ids=None,
                    races_bet=races_bet, total_bets=total_bets,
                    total_stake=total_stake, total_payout=total_payout,
                    hit_races=hit_races,
                )
                all_results.append(er)

                if er.races_bet >= 15 and er.roi > tt_best_roi:
                    tt_best_roi = er.roi
                    tt_best_label = f"conf≥{conf} EV≥{ev} cand≤{mc} kelly={kf}"

            if kelly_done % 10 == 0 or kelly_done == kelly_total:
                elapsed = time.time() - t0
                eta = elapsed / kelly_done * (kelly_total - kelly_done) if kelly_done > 0 else 0
                print(
                    f"  [{kelly_done}/{kelly_total}] {elapsed:.0f}s経過 残り{eta:.0f}s "
                    f"| {TT_LABELS[tt]} 暫定最良: ROI {tt_best_roi*100:.1f}% ({tt_best_label})",
                    flush=True,
                )

    print(f"  完了: {len(all_results)} 結果 ({time.time()-t0:.0f}s)")
    return all_results


# ── Step 3: Results analysis ─────────────────────────────────────────

def analyze_results(
    results: list[EvalResult],
    label: str,
    min_races: int = 15,
) -> dict[TicketType, EvalResult]:
    print(f"\n{'=' * 90}")
    print(f"  STEP 3: 結果分析 [{label}]")
    print("=" * 90)

    best: dict[TicketType, EvalResult] = {}

    for tt in TICKET_TYPES:
        tt_results = [r for r in results if r.ticket_type == tt and r.races_bet >= min_races]
        if not tt_results:
            print(f"\n  【{TT_LABELS[tt]}】 有効な結果なし")
            continue
        tt_results.sort(key=lambda r: r.roi, reverse=True)
        best[tt] = tt_results[0]

        print(f"\n  【{TT_LABELS[tt]}】 Top 5:")
        print(f"    {'#':>3} {'回収率':>8} {'損益':>10} {'投票R':>6} {'的中率':>8} {'投下額':>10}  パラメータ")
        print(f"    {'─' * 82}")
        for i, r in enumerate(tt_results[:5], 1):
            print(
                f"    {i:>3} {r.roi*100:>7.1f}% {r.profit:>+10,.0f} "
                f"{r.races_bet:>6} {r.hit_rate*100:>7.1f}% {r.total_stake:>10,.0f}  "
                f"conf≥{r.min_confidence:.2f} EV≥{r.min_ev:.2f} cand≤{r.max_candidates} kelly={r.fractional_kelly}"
            )

    return best


# ── Step 4: Track breakdown ──────────────────────────────────────────

def track_breakdown(
    caches: dict[int, RaceCache],
    best: dict[TicketType, EvalResult],
    date_start: date,
    date_end: date,
) -> dict[TicketType, dict]:
    print(f"\n{'=' * 90}")
    print(f"  STEP 4: トラック別分析")
    print("=" * 90)

    races = {k: v for k, v in caches.items() if date_start <= v.race_date <= date_end}
    track_ids = sorted(TRACK_NAMES.keys())
    analysis: dict[TicketType, dict] = {}

    for tt, best_er in best.items():
        per_track: dict[int, EvalResult] = {}
        for tid in track_ids:
            track_races = {k: v for k, v in races.items() if v.track_id == tid}
            er = _eval_on_races(
                track_races, tt, best_er.min_confidence, best_er.min_ev,
                best_er.max_candidates, best_er.fractional_kelly,
            )
            er.track_ids = [tid]
            per_track[tid] = er

        profitable = [tid for tid, er in per_track.items() if er.races_bet >= 3 and er.roi >= 1.0]
        subset_er = None
        if profitable:
            track_subset = {k: v for k, v in races.items() if v.track_id in profitable}
            subset_er = _eval_on_races(
                track_subset, tt, best_er.min_confidence, best_er.min_ev,
                best_er.max_candidates, best_er.fractional_kelly,
            )
            subset_er.track_ids = profitable

        analysis[tt] = {"per_track": per_track, "profitable": profitable, "subset": subset_er}

        print(f"\n  【{TT_LABELS[tt]}】 (conf≥{best_er.min_confidence} EV≥{best_er.min_ev} "
              f"cand≤{best_er.max_candidates} kelly={best_er.fractional_kelly})")
        print(f"    {'場名':<8} {'回収率':>8} {'損益':>10} {'投票R':>6} {'的中率':>8}")
        print(f"    {'─' * 46}")
        for tid in track_ids:
            er = per_track[tid]
            if er.races_bet == 0:
                continue
            mark = " ★" if tid in profitable else ""
            print(f"    {TRACK_NAMES[tid]:<8} {er.roi*100:>7.1f}% {er.profit:>+10,.0f} {er.races_bet:>6} {er.hit_rate*100:>7.1f}%{mark}")
        if subset_er and subset_er.races_bet > 0:
            names = "+".join(TRACK_NAMES[t] for t in profitable)
            print(f"    {'─' * 46}")
            print(f"    ★合計    {subset_er.roi*100:>7.1f}% {subset_er.profit:>+10,.0f} {subset_er.races_bet:>6} {subset_er.hit_rate*100:>7.1f}%  ({names})")

    return analysis


def _eval_on_races(
    races: dict[int, RaceCache],
    tt: TicketType,
    conf: float,
    ev: float,
    mc: int,
    kf: float,
) -> EvalResult:
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
        fractional_kelly=kf, track_ids=None,
        races_bet=races_bet, total_bets=total_bets,
        total_stake=total_stake, total_payout=total_payout,
        hit_races=hit_races,
    )


# ── Step 5: Holdout validation ───────────────────────────────────────

def holdout_validation(
    caches: dict[int, RaceCache],
    dev_best: dict[TicketType, EvalResult],
    track_analysis: dict[TicketType, dict],
) -> None:
    print(f"\n{'=' * 90}")
    print(f"  STEP 5: ホールドアウト検証（2026-03）")
    print("=" * 90)

    hold_races = {k: v for k, v in caches.items() if v.race_date >= HOLDOUT_START}
    print(f"  検証レース数: {len(hold_races)}")

    print(f"\n  {'券種':<8} {'場':>6} {'開発ROI':>8} {'開発R':>6} "
          f"{'検証ROI':>8} {'検証損益':>10} {'検証R':>6} {'検証的中':>8}")
    print(f"  {'─' * 82}")

    winners: list[str] = []

    for tt in TICKET_TYPES:
        if tt not in dev_best:
            continue
        b = dev_best[tt]
        ta = track_analysis.get(tt, {})
        profitable = ta.get("profitable", [])

        configs = [("全場", hold_races)]
        if profitable:
            filtered = {k: v for k, v in hold_races.items() if v.track_id in profitable}
            track_label = "+".join(TRACK_NAMES[t] for t in profitable)
            configs.append((track_label[:6], filtered))

        for track_label, races in configs:
            er = _eval_on_races(races, tt, b.min_confidence, b.min_ev, b.max_candidates, b.fractional_kelly)
            d_roi = f"{b.roi*100:.1f}%"
            h_roi = f"{er.roi*100:.1f}%" if er.races_bet > 0 else "N/A"
            h_hr = f"{er.hit_rate*100:.1f}%" if er.races_bet > 0 else "N/A"
            mark = " ◎" if er.races_bet >= 3 and er.roi >= 1.0 else ""
            print(
                f"  {TT_LABELS[tt]:<8} {track_label:>6} {d_roi:>8} {b.races_bet:>6} "
                f"{h_roi:>8} {er.profit:>+10,.0f} {er.races_bet:>6} {h_hr:>8}{mark}"
            )
            if mark:
                winners.append(
                    f"    {TT_LABELS[tt]} [{track_label}]: 検証ROI {er.roi*100:.1f}%, "
                    f"損益 {er.profit:+,.0f}円, {er.races_bet}R, 的中率 {er.hit_rate*100:.1f}%\n"
                    f"      params: conf≥{b.min_confidence} EV≥{b.min_ev} cand≤{b.max_candidates} kelly={b.fractional_kelly}"
                )

    print(f"\n{'─' * 90}")
    if winners:
        print("  ◎ ホールドアウトで回収率100%超:")
        for w in winners:
            print(w)
    else:
        print("  △ ホールドアウトで回収率100%超の設定なし → Phase 2 以降の改善が必要")
    print(f"{'=' * 90}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    caches = precompute(HOLDOUT_END)
    if not caches:
        print("データなし。")
        sys.exit(1)

    dev_count = sum(1 for v in caches.values() if v.race_date <= DEV_END)
    hold_count = sum(1 for v in caches.values() if v.race_date >= HOLDOUT_START)
    print(f"\n  開発: {dev_count}R | ホールドアウト: {hold_count}R")

    dev_results = grid_search(caches, FULL_START, DEV_END, "開発セット")
    dev_best = analyze_results(dev_results, "開発セット")

    if not dev_best:
        print("有効な結果なし。")
        sys.exit(1)

    track_data = track_breakdown(caches, dev_best, FULL_START, DEV_END)
    holdout_validation(caches, dev_best, track_data)

    total = time.time() - t_start
    print(f"総実行時間: {total:.0f}秒 ({total/60:.1f}分)")


if __name__ == "__main__":
    main()
