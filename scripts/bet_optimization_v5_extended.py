"""V012 extended strategy search with edge filter, prob_power, and place bets.

Adds V012 strategy axes:
- min_edge: market disagreement filter
- prob_power: power-transform probabilities to concentrate on high-confidence
- PLACE (複勝): new ticket type

Usage:
    PYTHONPATH=scripts:$PYTHONPATH uv run python scripts/bet_optimization_v5_extended.py --model-version v012
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from providence.domain.enums import TicketType
from providence.strategy.bankroll import normalize_to_stakes
from providence.strategy.candidates import build_candidates
from providence.strategy.kelly import optimize_kelly_fractions
from providence.strategy.normalize import flatten_ticket_probs
from providence.strategy.types import RecommendedBet, SettledTicketPayout, StrategyConfig
from strategy_grid_search_v2 import (
    DEV_END,
    DEV_START,
    HOLDOUT_END,
    HOLDOUT_START,
    MIN_BET_RACES,
    RaceCache,
    precompute,
)

LOG_INTERVAL_SEC = 300

WIN = TicketType.WIN
PLACE = TicketType.PLACE
WIDE = TicketType.WIDE
QUINELLA = TicketType.QUINELLA
EXACTA = TicketType.EXACTA

TICKET_TYPES = [WIN, PLACE, WIDE, QUINELLA, EXACTA]
TICKET_LABELS = {
    WIN: "単勝", PLACE: "複勝", WIDE: "ワイド",
    QUINELLA: "2連複", EXACTA: "2連単",
}

MIN_HIT_RATE_BY_TICKET = {
    WIN: 0.0625, PLACE: 0.15, WIDE: 0.0536,
    QUINELLA: 0.0179, EXACTA: 0.0089,
}


@dataclass(frozen=True)
class ExtSpec:
    ticket_type: TicketType
    min_confidence: float
    min_ev: float
    min_edge: float | None
    prob_power: float | None
    max_odds: float | None
    max_candidates: int
    fractional_kelly: float


@dataclass
class ExtResult:
    spec: ExtSpec
    total_stake: float
    total_payout: float
    profit: float
    races_bet: int
    hit_races: int
    monthly_roi: dict[str, float]

    @property
    def roi(self) -> float:
        return self.profit / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hit_races / self.races_bet if self.races_bet > 0 else 0.0


def build_config(spec: ExtSpec) -> StrategyConfig:
    return StrategyConfig(
        fractional_kelly=spec.fractional_kelly,
        min_expected_value=spec.min_ev,
        min_confidence=spec.min_confidence,
        max_candidates=spec.max_candidates,
        max_total_stake=10_000,
        min_edge=spec.min_edge,
        prob_power=spec.prob_power,
        max_odds=spec.max_odds,
        allowed_ticket_types=frozenset([spec.ticket_type]),
    )


def evaluate(spec: ExtSpec, caches: dict[int, RaceCache], start: date, end: date) -> ExtResult:
    config = build_config(spec)
    total_stake = 0.0
    total_payout = 0.0
    races_bet = 0
    hit_races = 0
    monthly: dict[str, dict] = defaultdict(lambda: {"stake": 0.0, "payout": 0.0})

    for cache in caches.values():
        if not (start <= cache.race_date <= end):
            continue
        probs = flatten_ticket_probs(cache.bundle.ticket_probs, cache.bundle.index_map)
        cands = build_candidates(probs, cache.market_odds, confidence_score=cache.confidence, config=config)
        if not cands:
            continue
        weights = optimize_kelly_fractions(candidates=cands, bundle=cache.bundle, weight_cap=1.0)
        weights *= config.fractional_kelly
        recs = [
            RecommendedBet(
                ticket_type=c.ticket_type, combination=c.combination,
                probability=c.probability, odds_value=c.odds_value,
                expected_value=c.expected_value, confidence_score=c.confidence_score,
                kelly_fraction=float(w), recommended_bet=0.0, stake_weight=float(w),
            )
            for c, w in zip(cands, weights) if w > 0
        ]
        bets = normalize_to_stakes(recs, config=config)
        stake = sum(b.recommended_bet for b in bets if b.recommended_bet > 0)
        if stake <= 0:
            continue
        races_bet += 1
        payout = 0.0
        hit = False
        for bet in bets:
            if bet.recommended_bet <= 0:
                continue
            for sp in cache.payouts:
                if sp.ticket_type == bet.ticket_type and sp.combination == bet.combination:
                    pay = bet.recommended_bet * sp.payout_value
                    payout += pay
                    hit = True
        total_stake += stake
        total_payout += payout
        if hit:
            hit_races += 1
        ym = cache.race_date.strftime("%Y-%m")
        monthly[ym]["stake"] += stake
        monthly[ym]["payout"] += payout

    monthly_roi = {}
    for ym, d in sorted(monthly.items()):
        monthly_roi[ym] = (d["payout"] - d["stake"]) / d["stake"] if d["stake"] > 0 else 0.0

    return ExtResult(
        spec=spec,
        total_stake=total_stake,
        total_payout=total_payout,
        profit=total_payout - total_stake,
        races_bet=races_bet,
        hit_races=hit_races,
        monthly_roi=monthly_roi,
    )


def passes(r: ExtResult) -> bool:
    if r.races_bet < 5:
        return False
    if r.roi <= 0:
        return False
    min_hit = MIN_HIT_RATE_BY_TICKET.get(r.spec.ticket_type, 0.01)
    if r.hit_rate < min_hit:
        return False
    breakdowns = sum(1 for roi in r.monthly_roi.values() if roi <= -0.50)
    return breakdowns <= 2


def specs_for_ticket(tt: TicketType) -> list[ExtSpec]:
    confs = [0.90, 0.93, 0.95, 0.97, 0.99]
    evs = [0.0, 0.05, 0.10, 0.20]
    edges = [None, 0.02, 0.05, 0.10]
    powers = [None, 2.0, 3.0, 4.0]
    if tt == WIN:
        max_odds_vals = [5.0, 10.0, None]
        cands = [3, 5]
        kellys = [0.05, 0.10]
    elif tt == PLACE:
        max_odds_vals = [3.0, 5.0, None]
        cands = [3, 5]
        kellys = [0.10, 0.25]
    elif tt == WIDE:
        max_odds_vals = [30.0, 50.0, None]
        cands = [2, 3]
        kellys = [0.10, 0.25]
    else:
        max_odds_vals = [20.0, 50.0, None]
        cands = [2, 3, 5]
        kellys = [0.10, 0.25]

    specs = []
    for conf, ev, edge, power, mo, cand, kelly in itertools.product(
        confs, evs, edges, powers, max_odds_vals, cands, kellys
    ):
        specs.append(ExtSpec(tt, conf, ev, edge, power, mo, cand, kelly))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", default="v012")
    args = parser.parse_args()

    print("=" * 90)
    print(f"  V012 拡張戦略探索 (model={args.model_version})")
    print("=" * 90)

    caches = precompute(args.model_version)
    if not caches:
        raise SystemExit("データなし")

    dev_races = {k: v for k, v in caches.items() if DEV_START <= v.race_date <= DEV_END}
    holdout_races = {k: v for k, v in caches.items() if HOLDOUT_START <= v.race_date <= HOLDOUT_END}
    print(f"  開発: {len(dev_races)}R | ホールドアウト: {len(holdout_races)}R", flush=True)

    summary: dict[str, Any] = {
        "model_version": args.model_version,
        "search_type": "v5_extended",
        "dev_period": [DEV_START.isoformat(), DEV_END.isoformat()],
        "holdout_period": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "tickets": {},
    }

    overall_best: list[dict] = []

    for ticket_type in TICKET_TYPES:
        label = TICKET_LABELS[ticket_type]
        specs = specs_for_ticket(ticket_type)
        print(f"\n{'='*90}")
        print(f"  券種: {label} ({len(specs)} specs)")
        print(f"{'='*90}", flush=True)

        last_log = time.time()
        dev_valid: list[ExtResult] = []

        for i, spec in enumerate(specs):
            result = evaluate(spec, dev_races, DEV_START, DEV_END)
            if passes(result):
                dev_valid.append(result)

            if time.time() - last_log > LOG_INTERVAL_SEC:
                best_str = "なし"
                if dev_valid:
                    b = max(dev_valid, key=lambda r: r.profit)
                    best_str = f"ROI={b.roi*100:+.1f}% profit={b.profit:+,.0f}円 bet={b.races_bet}"
                elapsed = (time.time() - last_log) + LOG_INTERVAL_SEC
                print(
                    f"  [{(time.time()-last_log+LOG_INTERVAL_SEC):.0f}s] {label}: {i+1}/{len(specs)} "
                    f"valid={len(dev_valid)} best=[{best_str}]",
                    flush=True,
                )
                last_log = time.time()

        print(f"  {label}: 開発有効={len(dev_valid)}", flush=True)

        dev_valid.sort(key=lambda r: r.profit, reverse=True)
        top_dev = dev_valid[:20]

        holdout_results = []
        for candidate in top_dev:
            h = evaluate(candidate.spec, holdout_races, HOLDOUT_START, HOLDOUT_END)
            holdout_results.append({"dev": candidate, "holdout": h})

        holdout_results.sort(key=lambda r: r["holdout"].profit, reverse=True)

        ticket_summary: dict[str, Any] = {
            "total_specs": len(specs),
            "dev_valid": len(dev_valid),
            "top_dev": [],
            "holdout": [],
        }

        for r in top_dev[:5]:
            ticket_summary["top_dev"].append({
                "spec": {
                    "conf": r.spec.min_confidence, "ev": r.spec.min_ev,
                    "edge": r.spec.min_edge, "power": r.spec.prob_power,
                    "max_odds": r.spec.max_odds, "cand": r.spec.max_candidates,
                    "kelly": r.spec.fractional_kelly,
                },
                "roi": r.roi, "profit": r.profit, "bet": r.races_bet,
                "hit_rate": r.hit_rate,
            })

        for row in holdout_results[:5]:
            h = row["holdout"]
            d = row["dev"]
            ticket_summary["holdout"].append({
                "spec": {
                    "conf": h.spec.min_confidence, "ev": h.spec.min_ev,
                    "edge": h.spec.min_edge, "power": h.spec.prob_power,
                    "max_odds": h.spec.max_odds, "cand": h.spec.max_candidates,
                    "kelly": h.spec.fractional_kelly,
                },
                "dev_roi": d.roi, "dev_profit": d.profit,
                "holdout_roi": h.roi, "holdout_profit": h.profit,
                "holdout_bet": h.races_bet, "holdout_hit_rate": h.hit_rate,
            })
            if h.profit > 0:
                overall_best.append({
                    "ticket": label,
                    "spec": {
                        "conf": h.spec.min_confidence, "ev": h.spec.min_ev,
                        "edge": h.spec.min_edge, "power": h.spec.prob_power,
                        "max_odds": h.spec.max_odds, "cand": h.spec.max_candidates,
                        "kelly": h.spec.fractional_kelly,
                    },
                    "dev_roi": d.roi, "holdout_roi": h.roi,
                    "holdout_profit": h.profit, "holdout_bet": h.races_bet,
                })

        summary["tickets"][label] = ticket_summary

        if holdout_results and holdout_results[0]["holdout"].profit > 0:
            h = holdout_results[0]["holdout"]
            print(
                f"  ★ {label} holdout: ROI={h.roi*100:+.1f}% profit={h.profit:+,.0f}円 bet={h.races_bet}",
                flush=True,
            )
        else:
            best_h = holdout_results[0]["holdout"] if holdout_results else None
            if best_h:
                print(f"  {label} holdout best: ROI={best_h.roi*100:+.1f}% profit={best_h.profit:+,.0f}円", flush=True)
            else:
                print(f"  {label}: holdout候補なし", flush=True)

    print(f"\n{'='*90}")
    print("  全体サマリ")
    print(f"{'='*90}")
    if overall_best:
        for item in overall_best:
            print(f"  ★ {item['ticket']}: holdout ROI={item['holdout_roi']*100:+.1f}% profit={item['holdout_profit']:+,.0f}円", flush=True)
    else:
        print("  holdout利益プラスの券種: なし", flush=True)

    summary["overall_best"] = overall_best

    out_dir = Path("data/strategy_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{args.model_version}_v5_extended.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  結果保存: {out_path}", flush=True)


if __name__ == "__main__":
    main()
