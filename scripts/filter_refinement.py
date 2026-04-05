"""Filter refinement analysis for Win and Wide tickets.

Step 1a: Monthly ROI trends
Step 1b: 6-axis dimensional ROI analysis
Step 2:  Filter search (select/exclude, all 2-axis combos)
Step 3:  Holdout validation
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
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
from providence.probability.plackett_luce import compute_win_probs
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
)

DEV_START = date(2025, 8, 8)
DEV_END = date(2026, 2, 28)
HOLD_START = date(2026, 3, 1)
HOLD_END = date(2026, 3, 31)
JC = dt_time(15, 0, 0)

TRACK_NAMES = {tc.value: tc.japanese_name for tc in TrackCode}

CONFIGS = {
    TicketType.WIN: StrategyConfig(
        min_confidence=0.95, min_expected_value=0.3,
        max_candidates=5, fractional_kelly=0.1,
        max_total_stake=10_000,
        allowed_ticket_types=frozenset({TicketType.WIN}),
    ),
    TicketType.WIDE: StrategyConfig(
        min_confidence=0.8, min_expected_value=0.3,
        max_candidates=3, fractional_kelly=0.1,
        max_total_stake=10_000,
        allowed_ticket_types=frozenset({TicketType.WIDE}),
    ),
}

MIN_R_2AXIS = 30
MIN_R_HOLDOUT = 15


@dataclass
class RaceCtx:
    race_id: int
    bundle: RacePredictionBundle
    market_odds: list[MarketTicketOdds]
    payouts: list[SettledTicketPayout]
    judgment_time: datetime
    confidence: float
    race_date: date
    track_id: int
    weather: str
    track_condition: str
    race_number: int
    field_size: int
    model_top1_pp: int
    market_fav_pp: int


@dataclass
class Metrics:
    races_bet: int = 0
    total_bets: int = 0
    total_stake: float = 0.0
    total_payout: float = 0.0
    hit_races: int = 0

    @property
    def roi(self) -> float:
        return self.total_payout / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def profit(self) -> float:
        return self.total_payout - self.total_stake

    @property
    def hit_rate(self) -> float:
        return self.hit_races / self.races_bet if self.races_bet > 0 else 0.0


def _classify_weather(w: str | None) -> str:
    if w is None:
        return "晴"
    if w in ("晴",):
        return "晴"
    if w in ("曇",):
        return "曇"
    return "雨系"


def _classify_condition(c: str | None) -> str:
    if c is None or c == "良":
        return "良"
    if c == "湿":
        return "湿"
    return "斑"


# ── Data loading ─────────────────────────────────────────────────────

def load_data() -> dict[int, RaceCtx]:
    print("=" * 90)
    print("  データ読み込み + 予測")
    print("=" * 90)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    ms = ModelStore()
    repo = Repository()
    sf = get_session_factory()

    raw_df = loader.load_race_dataset(end_date=HOLD_END)
    ck = FeaturePipeline.cache_key({
        "purpose": "filter", "rows": len(raw_df),
        "race_min": raw_df["race_id"].min(), "race_max": raw_df["race_id"].max(),
        "entry_max": raw_df["race_entry_id"].max(),
        "date_min": raw_df["race_date"].min(), "date_max": raw_df["race_date"].max(),
        "end_date": HOLD_END.isoformat(),
    })
    cp = Path("data/processed") / f"filter_features_{ck}.parquet"
    feats = pipeline.build_and_cache(raw_df, str(cp))
    ds = feats.filter(pl.col("race_date").is_between(DEV_START, HOLD_END, closed="both"))

    rr = (
        ds.select(["race_id", "race_date", "track_id", "race_number"])
        .unique(maintain_order=True)
        .sort(["race_date", "race_number", "race_id"])
    )
    dates = sorted(set(r["race_date"] for r in rr.iter_rows(named=True)))
    total = rr.height
    print(f"  {len(dates)}日, {total}レース", flush=True)

    predictor = Predictor(ms, pipeline, loader, version="v003")
    ctxs: dict[int, RaceCtx] = {}
    done = 0

    from sqlalchemy import text

    for rd in dates:
        day_f = ds.filter(pl.col("race_date") == rd).sort(
            ["race_number", "race_id", "post_position"]
        )
        bundles = predictor.predict_feature_races(day_f)
        rids = list(bundles)
        jt = datetime.combine(rd, JC)

        with sf() as session:
            mkt_by = repo.get_latest_market_odds_for_races(session, rids, judgment_time=jt)
            pay_by = repo.get_ticket_payouts_for_races(session, rids)

            weather_cond = {}
            if rids:
                placeholders = ",".join(str(r) for r in rids)
                rows = session.execute(text(
                    f"SELECT id, weather, track_condition FROM races WHERE id IN ({placeholders})"
                )).fetchall()
                for r in rows:
                    weather_cond[r[0]] = (r[1], r[2])

        for rid, bundle in bundles.items():
            done += 1
            odds = market_odds_from_rows(mkt_by.get(rid, []))
            payouts = payouts_from_rows(pay_by.get(rid, []))
            if not odds or not payouts:
                continue

            meta = rr.filter(pl.col("race_id") == rid).row(0, named=True)
            w_raw, c_raw = weather_cond.get(rid, (None, None))
            conf = race_confidence(bundle)

            scores = bundle.scores
            win_probs = compute_win_probs(scores, bundle.temperature)
            model_top1_idx = int(np.argmax(win_probs))
            model_top1_pp = bundle.index_map.post_position_for_index(model_top1_idx)

            win_odds = {o.combination: o.odds_value for o in odds if o.ticket_type == TicketType.WIN}
            market_fav_pp = 0
            min_odds = float("inf")
            for combo, ov in win_odds.items():
                if ov < min_odds:
                    min_odds = ov
                    market_fav_pp = combo[0]

            fs = len(scores)

            ctxs[rid] = RaceCtx(
                race_id=rid, bundle=bundle, market_odds=odds, payouts=payouts,
                judgment_time=jt, confidence=conf, race_date=rd,
                track_id=int(meta["track_id"]),
                weather=_classify_weather(w_raw),
                track_condition=_classify_condition(c_raw),
                race_number=int(meta["race_number"]),
                field_size=fs,
                model_top1_pp=model_top1_pp,
                market_fav_pp=market_fav_pp,
            )

        if done % 500 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    print(f"  完了: {len(ctxs)} 有効レース")
    return ctxs


# ── Kelly evaluation ─────────────────────────────────────────────────

def evaluate(
    ctxs: dict[int, RaceCtx],
    tt: TicketType,
    config: StrategyConfig,
    race_filter=None,
) -> Metrics:
    m = Metrics()
    stake_cfg = StrategyConfig(
        fractional_kelly=config.fractional_kelly,
        max_total_stake=config.max_total_stake,
    )
    for ctx in ctxs.values():
        if race_filter and not race_filter(ctx):
            continue
        if ctx.confidence < config.min_confidence:
            continue
        predicted = flatten_ticket_probs(ctx.bundle.ticket_probs, ctx.bundle.index_map)
        candidates = build_candidates(
            predicted, ctx.market_odds,
            confidence_score=ctx.confidence, config=config,
        )
        if not candidates:
            continue
        weights = optimize_kelly_fractions(candidates=candidates, bundle=ctx.bundle)
        if np.all(weights <= 0):
            continue
        scaled = weights * config.fractional_kelly
        recs = [
            RecommendedBet(
                ticket_type=c.ticket_type, combination=c.combination,
                probability=c.probability, odds_value=c.odds_value,
                expected_value=c.expected_value, confidence_score=c.confidence_score,
                kelly_fraction=float(w), recommended_bet=0.0, stake_weight=float(w),
            )
            for c, w in zip(candidates, scaled) if w > 0
        ]
        rounded = normalize_to_stakes(recs, config=stake_cfg)
        if not rounded:
            continue
        settled = settle_recommendations(rounded, ctx.payouts)
        m.races_bet += 1
        m.total_bets += len(settled)
        m.total_stake += sum(s.recommendation.recommended_bet for s in settled)
        m.total_payout += sum(s.payout_amount for s in settled)
        if any(s.hit for s in settled):
            m.hit_races += 1
    return m


# ── Step 1a: Monthly ROI ─────────────────────────────────────────────

def step_1a(dev: dict[int, RaceCtx]) -> None:
    print(f"\n{'=' * 90}")
    print("  Step 1a: 月別ROI推移")
    print("=" * 90)

    months = sorted(set(c.race_date.strftime("%Y-%m") for c in dev.values()))

    for tt, config in CONFIGS.items():
        print(f"\n  【{tt.value}】 conf≥{config.min_confidence} EV≥{config.min_expected_value}")
        print(f"    {'月':>8} {'ROI':>8} {'損益':>10} {'投票R':>6} {'的中率':>8}")
        print(f"    {'─' * 48}")
        for month in months:
            m = evaluate(dev, tt, config,
                         race_filter=lambda c, mo=month: c.race_date.strftime("%Y-%m") == mo)
            if m.races_bet == 0:
                print(f"    {month:>8} {'N/A':>8}")
                continue
            print(f"    {month:>8} {m.roi*100:>7.1f}% {m.profit:>+10,.0f} {m.races_bet:>6} {m.hit_rate*100:>7.1f}%")


# ── Step 1b: 6-axis analysis ─────────────────────────────────────────

def step_1b(dev: dict[int, RaceCtx]) -> dict[str, dict]:
    print(f"\n{'=' * 90}")
    print("  Step 1b: 6軸次元別ROI分析")
    print("=" * 90)

    axes = {
        "トラック": lambda c: TRACK_NAMES.get(c.track_id, str(c.track_id)),
        "天候": lambda c: c.weather,
        "走路": lambda c: c.track_condition,
        "出走数": lambda c: str(c.field_size),
        "R番号帯": lambda c: "前半R1-6" if c.race_number <= 6 else "後半R7-12",
        "モデルvs市場": lambda c: "一致" if c.model_top1_pp == c.market_fav_pp else "不一致",
    }

    all_results: dict[str, dict] = {}

    for tt, config in CONFIGS.items():
        tt_results: dict[str, list[tuple[str, Metrics]]] = {}
        print(f"\n  【{tt.value}】")

        for axis_name, axis_fn in axes.items():
            values = sorted(set(axis_fn(c) for c in dev.values()))
            axis_metrics: list[tuple[str, Metrics]] = []

            print(f"\n    ── {axis_name} ──")
            print(f"    {'値':<12} {'ROI':>8} {'損益':>10} {'投票R':>6} {'的中率':>8}")
            print(f"    {'─' * 52}")

            for val in values:
                m = evaluate(dev, tt, config,
                             race_filter=lambda c, fn=axis_fn, v=val: fn(c) == v)
                axis_metrics.append((val, m))
                if m.races_bet == 0:
                    print(f"    {val:<12} {'N/A':>8}")
                    continue
                mark = " ★" if m.races_bet >= 30 and m.roi >= 1.0 else ""
                mark2 = " ▼" if m.races_bet >= 30 and m.roi < 0.85 else ""
                print(
                    f"    {val:<12} {m.roi*100:>7.1f}% {m.profit:>+10,.0f} "
                    f"{m.races_bet:>6} {m.hit_rate*100:>7.1f}%{mark}{mark2}"
                )

            tt_results[axis_name] = axis_metrics

        all_results[tt.value] = tt_results

    return all_results


# ── Step 2: Filter search ────────────────────────────────────────────

@dataclass
class FilterCandidate:
    tt: TicketType
    filter_type: str
    description: str
    dev_metrics: Metrics
    filter_fn: object


def step_2(
    dev: dict[int, RaceCtx],
    axis_results: dict[str, dict],
) -> list[FilterCandidate]:
    print(f"\n{'=' * 90}")
    print("  Step 2: フィルタ探索")
    print("=" * 90)

    axes_fns = {
        "トラック": lambda c: TRACK_NAMES.get(c.track_id, str(c.track_id)),
        "天候": lambda c: c.weather,
        "走路": lambda c: c.track_condition,
        "出走数": lambda c: str(c.field_size),
        "R番号帯": lambda c: "前半R1-6" if c.race_number <= 6 else "後半R7-12",
        "モデルvs市場": lambda c: "一致" if c.model_top1_pp == c.market_fav_pp else "不一致",
    }

    # Axis pairs for 2-axis (exclude weather x track_condition)
    axis_names = list(axes_fns.keys())
    axis_pairs = [
        (a, b) for a, b in itertools.combinations(axis_names, 2)
        if not ({a, b} == {"天候", "走路"})
    ]

    all_candidates: list[FilterCandidate] = []

    for tt, config in CONFIGS.items():
        print(f"\n  【{tt.value}】")
        tt_candidates: list[FilterCandidate] = []

        # 1-axis select filters
        for axis_name, axis_fn in axes_fns.items():
            values = sorted(set(axis_fn(c) for c in dev.values()))
            for val in values:
                fn = lambda c, afn=axis_fn, v=val: afn(c) == v
                m = evaluate(dev, tt, config, race_filter=fn)
                if m.races_bet >= MIN_R_2AXIS:
                    tt_candidates.append(FilterCandidate(
                        tt=tt, filter_type="選択",
                        description=f"{axis_name}={val}",
                        dev_metrics=m, filter_fn=fn,
                    ))

        # 1-axis exclude filters
        for axis_name, axis_fn in axes_fns.items():
            values = sorted(set(axis_fn(c) for c in dev.values()))
            for val in values:
                fn = lambda c, afn=axis_fn, v=val: afn(c) != v
                m = evaluate(dev, tt, config, race_filter=fn)
                if m.races_bet >= MIN_R_2AXIS:
                    tt_candidates.append(FilterCandidate(
                        tt=tt, filter_type="除外",
                        description=f"{axis_name}≠{val}",
                        dev_metrics=m, filter_fn=fn,
                    ))

        # 2-axis combinations (all pairs, all value combos)
        for ax_a, ax_b in axis_pairs:
            fn_a = axes_fns[ax_a]
            fn_b = axes_fns[ax_b]
            vals_a = sorted(set(fn_a(c) for c in dev.values()))
            vals_b = sorted(set(fn_b(c) for c in dev.values()))

            for va, vb in itertools.product(vals_a, vals_b):
                fn = lambda c, fa=fn_a, fb=fn_b, a=va, b=vb: fa(c) == a and fb(c) == b
                m = evaluate(dev, tt, config, race_filter=fn)
                if m.races_bet >= MIN_R_2AXIS:
                    tt_candidates.append(FilterCandidate(
                        tt=tt, filter_type="2軸",
                        description=f"{ax_a}={va} & {ax_b}={vb}",
                        dev_metrics=m, filter_fn=fn,
                    ))

        # Sort by ROI and print top results
        tt_candidates.sort(key=lambda f: f.dev_metrics.roi, reverse=True)
        top_n = [f for f in tt_candidates if f.dev_metrics.roi >= 1.0][:20]

        print(f"\n    開発ROI 100%超のフィルタ ({len(top_n)}件):")
        print(f"    {'#':>3} {'種別':>4} {'ROI':>8} {'損益':>10} {'投票R':>6} {'的中率':>8}  条件")
        print(f"    {'─' * 76}")
        for i, f in enumerate(top_n, 1):
            m = f.dev_metrics
            print(
                f"    {i:>3} {f.filter_type:>4} {m.roi*100:>7.1f}% {m.profit:>+10,.0f} "
                f"{m.races_bet:>6} {m.hit_rate*100:>7.1f}%  {f.description}"
            )

        all_candidates.extend(top_n)

    return all_candidates


# ── Step 3: Holdout validation ───────────────────────────────────────

def step_3(
    hold: dict[int, RaceCtx],
    candidates: list[FilterCandidate],
) -> None:
    print(f"\n{'=' * 90}")
    print("  Step 3: ホールドアウト検証 (2026-03)")
    print("=" * 90)

    for tt, config in CONFIGS.items():
        print(f"\n  【{tt.value}】")

        # Baseline
        base = evaluate(hold, tt, config)
        print(f"    ベースライン（フィルタなし）: ROI {base.roi*100:.1f}%, 損益 {base.profit:+,.0f}円, {base.races_bet}R")

        tt_cands = [c for c in candidates if c.tt == tt]
        print(f"\n    {'#':>3} {'種別':>4} {'開発ROI':>8} {'検証ROI':>8} {'検証損益':>10} {'検証R':>6} {'検証的中':>8}  条件")
        print(f"    {'─' * 88}")

        for i, cand in enumerate(tt_cands, 1):
            m = evaluate(hold, tt, config, race_filter=cand.filter_fn)
            mark = " ◎" if m.races_bet >= MIN_R_HOLDOUT and m.roi >= 1.0 else ""
            h_roi = f"{m.roi*100:.1f}%" if m.races_bet > 0 else "N/A"
            h_hr = f"{m.hit_rate*100:.1f}%" if m.races_bet > 0 else "N/A"
            print(
                f"    {i:>3} {cand.filter_type:>4} {cand.dev_metrics.roi*100:>7.1f}% "
                f"{h_roi:>8} {m.profit:>+10,.0f} {m.races_bet:>6} {h_hr:>8}  {cand.description}{mark}"
            )


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    ctxs = load_data()
    dev = {k: v for k, v in ctxs.items() if v.race_date <= DEV_END}
    hold = {k: v for k, v in ctxs.items() if v.race_date >= HOLD_START}
    print(f"  開発: {len(dev)}R | ホールドアウト: {len(hold)}R")

    step_1a(dev)
    axis_results = step_1b(dev)
    candidates = step_2(dev, axis_results)
    step_3(hold, candidates)

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")


if __name__ == "__main__":
    main()
