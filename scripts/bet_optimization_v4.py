"""Two-stage bet optimization for return maximization.

Focuses on v010 and explores untested strategy axes:
- min_probability_by_ticket on/off
- min_odds / max_odds
- confidence / EV local search
- fine search for max_candidates / fractional_kelly on multi-candidate tickets

Outputs periodic progress every 10 minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from providence.domain.enums import TicketType
from providence.strategy.bankroll import normalize_to_stakes
from providence.strategy.candidates import build_candidates
from providence.strategy.kelly import optimize_kelly_fractions
from providence.strategy.normalize import flatten_ticket_probs
from providence.strategy.types import RecommendedBet, StrategyConfig
from strategy_grid_search_v2 import (
    DEV_END,
    DEV_START,
    HOLDOUT_END,
    HOLDOUT_START,
    MIN_BET_RACES,
    RaceCache,
    precompute,
)

LOG_INTERVAL_SEC = 600
TOP_SEEDS = 5
TOP_FINE = 10

WIN = TicketType.WIN
WIDE = TicketType.WIDE
QUINELLA = TicketType.QUINELLA
EXACTA = TicketType.EXACTA

TICKET_TYPES = [WIN, WIDE, QUINELLA, EXACTA]
TICKET_LABELS = {WIN: "単勝", WIDE: "ワイド", QUINELLA: "2連複", EXACTA: "2連単"}

MIN_HIT_RATE_BY_TICKET = {
    WIN: 0.0625,
    WIDE: 0.0536,
    QUINELLA: 0.0179,
    EXACTA: 0.0089,
}

WIN_CONF = [0.90, 0.93, 0.95, 0.97]
WIN_EV = [0.0, 0.05, 0.10, 0.20]
WIN_MIN_ODDS = [1.5, 2.0, None]
WIN_MAX_ODDS = [5.0, 8.0, 10.0, None]
WIN_MIN_PROB = [None, 0.03, 0.05, 0.07]

MULTI_CONF = [0.93, 0.95, 0.97, 0.99]
MULTI_EV = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
WIDE_MAX_ODDS = [20.0, 30.0, 50.0, None]
PAIR_MAX_ODDS = [10.0, 20.0, 30.0, 50.0, None]
WIDE_MIN_PROB = [None, 0.03, 0.05, 0.07]
QUINELLA_MIN_PROB = [None, 0.01, 0.02, 0.03]
EXACTA_MIN_PROB = [None, 0.005, 0.01, 0.02]
MAX_CANDIDATES = [2, 3, 5]
KELLY_VALS = [0.05, 0.10, 0.25]

FOLD_RANGES = {
    "A": (date(2025, 5, 1), date(2025, 7, 31)),
    "B": (date(2025, 8, 1), date(2025, 10, 31)),
    "C": (date(2025, 11, 1), date(2025, 12, 31)),
}


@dataclass(frozen=True)
class SearchSpec:
    ticket_type: TicketType
    min_confidence: float
    min_ev: float
    min_probability: float | None
    min_odds: float | None
    max_odds: float | None
    max_candidates: int
    fractional_kelly: float


@dataclass
class EvalResult:
    spec: SearchSpec
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

    @property
    def monthly_breakdowns(self) -> int:
        return sum(1 for roi in self.monthly_roi.values() if roi <= -0.30)


class ProgressLogger:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.last_logged_at = self.started_at

    def maybe_log(
        self,
        *,
        phase: str,
        ticket_label: str,
        done: int,
        total: int,
        best: EvalResult | None,
        valid_candidates: int,
    ) -> None:
        now = time.time()
        if done != total and (now - self.last_logged_at) < LOG_INTERVAL_SEC:
            return
        self.last_logged_at = now
        elapsed = now - self.started_at
        eta = elapsed / done * (total - done) if done > 0 else 0.0
        print(f"[{elapsed/60:.1f}min] {phase} / {ticket_label}: {done}/{total} 完了", flush=True)
        print(f"  valid_candidates={valid_candidates}", flush=True)
        if best is not None:
            print(
                "  best: "
                f"conf>={best.spec.min_confidence} EV>={best.spec.min_ev} "
                f"min_prob={best.spec.min_probability} min_odds={best.spec.min_odds} max_odds={best.spec.max_odds} "
                f"cand<={best.spec.max_candidates} kelly={best.spec.fractional_kelly}",
                flush=True,
            )
            print(
                f"  profit={best.profit:+,.0f}円, bet_races={best.races_bet}, ROI={best.roi*100:+.1f}%",
                flush=True,
            )
        print(f"  ETA: {eta/60:.1f}min", flush=True)


def build_strategy_config(spec: SearchSpec) -> StrategyConfig:
    min_prob_map = None
    if spec.min_probability is not None:
        min_prob_map = {spec.ticket_type: spec.min_probability}
    return StrategyConfig(
        allowed_ticket_types=frozenset({spec.ticket_type}),
        min_confidence=spec.min_confidence,
        min_expected_value=spec.min_ev,
        min_probability_by_ticket=min_prob_map,
        min_odds=spec.min_odds,
        max_odds=spec.max_odds,
        max_candidates=spec.max_candidates,
        fractional_kelly=spec.fractional_kelly,
        max_total_stake=10_000,
    )


def evaluate_spec(races: dict[int, RaceCache], spec: SearchSpec) -> EvalResult:
    config = build_strategy_config(spec)
    monthly = defaultdict(lambda: {"stake": 0.0, "payout": 0.0})
    total_stake = 0.0
    total_payout = 0.0
    races_bet = 0
    hit_races = 0

    for rc in races.values():
        if rc.confidence < config.min_confidence:
            continue
        predicted = flatten_ticket_probs(rc.bundle.ticket_probs, rc.bundle.index_map)
        candidates = build_candidates(predicted, rc.market_odds, confidence_score=rc.confidence, config=config)
        if not candidates:
            continue
        weights = optimize_kelly_fractions(candidates=candidates, bundle=rc.bundle)
        if all(weight <= 0 for weight in weights):
            continue
        scaled = weights * config.fractional_kelly
        recs = [
            RecommendedBet(
                ticket_type=c.ticket_type,
                combination=c.combination,
                probability=c.probability,
                odds_value=c.odds_value,
                expected_value=c.expected_value,
                confidence_score=c.confidence_score,
                kelly_fraction=float(w),
                recommended_bet=0.0,
                stake_weight=float(w),
            )
            for c, w in zip(candidates, scaled, strict=False)
            if w > 0
        ]
        rounded = normalize_to_stakes(recs, config=config)
        if not rounded:
            continue
        # settlement
        settled = []
        for rec in rounded:
            hit_row = next(
                (
                    p
                    for p in rc.payouts
                    if p.ticket_type == rec.ticket_type and p.combination == rec.combination
                ),
                None,
            )
            payout_amount = rec.recommended_bet * hit_row.payout_value if hit_row else 0.0
            settled.append((rec, payout_amount, bool(hit_row)))
        stake = sum(rec.recommended_bet for rec, _, _ in settled)
        payout = sum(payout for _, payout, _ in settled)
        month = rc.race_date.strftime("%Y-%m")
        monthly[month]["stake"] += stake
        monthly[month]["payout"] += payout
        total_stake += stake
        total_payout += payout
        races_bet += 1
        if any(hit for _, _, hit in settled):
            hit_races += 1

    monthly_roi = {
        month: ((vals["payout"] - vals["stake"]) / vals["stake"]) if vals["stake"] > 0 else 0.0
        for month, vals in monthly.items()
    }
    return EvalResult(
        spec=spec,
        total_stake=total_stake,
        total_payout=total_payout,
        profit=total_payout - total_stake,
        races_bet=races_bet,
        hit_races=hit_races,
        monthly_roi=monthly_roi,
    )


def passes_constraints(result: EvalResult) -> bool:
    if result.total_stake <= 0:
        return False
    if result.roi <= 0:
        return False
    if result.races_bet < MIN_BET_RACES:
        return False
    if result.hit_rate < MIN_HIT_RATE_BY_TICKET[result.spec.ticket_type]:
        return False
    if result.monthly_breakdowns >= 2:
        return False
    return True


def pareto_front(results: list[EvalResult]) -> list[EvalResult]:
    front: list[EvalResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if (
                other.profit >= r.profit
                and other.races_bet >= r.races_bet
                and (other.profit > r.profit or other.races_bet > r.races_bet)
            ):
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda x: (x.profit, x.races_bet, x.roi), reverse=True)
    return front


def neighbors(values: list[Any], current: Any) -> list[Any]:
    idx = values.index(current)
    out = {current}
    if idx > 0:
        out.add(values[idx - 1])
    if idx < len(values) - 1:
        out.add(values[idx + 1])
    return list(out)


def coarse_specs_for_ticket(ticket_type: TicketType) -> list[SearchSpec]:
    specs: list[SearchSpec] = []
    if ticket_type == WIN:
        for conf, ev, min_odds, max_odds, min_prob in itertools.product(
            WIN_CONF, WIN_EV, WIN_MIN_ODDS, WIN_MAX_ODDS, WIN_MIN_PROB
        ):
            if min_odds is not None and max_odds is not None and min_odds >= max_odds:
                continue
            specs.append(
                SearchSpec(ticket_type, conf, ev, min_prob, min_odds, max_odds, 5, 0.05)
            )
        return specs

    max_odds_vals = WIDE_MAX_ODDS if ticket_type == WIDE else PAIR_MAX_ODDS
    min_probs = (
        WIDE_MIN_PROB if ticket_type == WIDE else QUINELLA_MIN_PROB if ticket_type == QUINELLA else EXACTA_MIN_PROB
    )
    base_cand = 2 if ticket_type in {WIDE, EXACTA} else 5
    base_kelly = 0.25 if ticket_type in {WIDE, EXACTA} else 0.10
    for conf, ev, max_odds, min_prob in itertools.product(MULTI_CONF, MULTI_EV, max_odds_vals, min_probs):
        specs.append(
            SearchSpec(ticket_type, conf, ev, min_prob, None, max_odds, base_cand, base_kelly)
        )
    return specs


def fine_specs_from_seeds(ticket_type: TicketType, seeds: list[EvalResult]) -> list[SearchSpec]:
    specs: set[SearchSpec] = set()
    for seed in seeds[:TOP_SEEDS]:
        s = seed.spec
        if ticket_type == WIN:
            for conf, ev in itertools.product(neighbors(WIN_CONF, s.min_confidence), neighbors(WIN_EV, s.min_ev)):
                specs.add(SearchSpec(ticket_type, conf, ev, s.min_probability, s.min_odds, s.max_odds, s.max_candidates, s.fractional_kelly))
            for min_prob, min_odds, max_odds in itertools.product(
                neighbors(WIN_MIN_PROB, s.min_probability),
                neighbors(WIN_MIN_ODDS, s.min_odds),
                neighbors(WIN_MAX_ODDS, s.max_odds),
            ):
                if min_odds is not None and max_odds is not None and min_odds >= max_odds:
                    continue
                specs.add(SearchSpec(ticket_type, s.min_confidence, s.min_ev, min_prob, min_odds, max_odds, s.max_candidates, s.fractional_kelly))
        else:
            conf_vals = neighbors(MULTI_CONF, s.min_confidence)
            ev_vals = neighbors(MULTI_EV, s.min_ev)
            odds_vals = neighbors(WIDE_MAX_ODDS if ticket_type == WIDE else PAIR_MAX_ODDS, s.max_odds)
            prob_vals = neighbors(
                WIDE_MIN_PROB if ticket_type == WIDE else QUINELLA_MIN_PROB if ticket_type == QUINELLA else EXACTA_MIN_PROB,
                s.min_probability,
            )
            for conf, ev in itertools.product(conf_vals, ev_vals):
                specs.add(SearchSpec(ticket_type, conf, ev, s.min_probability, None, s.max_odds, s.max_candidates, s.fractional_kelly))
            for max_odds, min_prob in itertools.product(odds_vals, prob_vals):
                specs.add(SearchSpec(ticket_type, s.min_confidence, s.min_ev, min_prob, None, max_odds, s.max_candidates, s.fractional_kelly))
            for cand, kelly in itertools.product(MAX_CANDIDATES, KELLY_VALS):
                specs.add(SearchSpec(ticket_type, s.min_confidence, s.min_ev, s.min_probability, None, s.max_odds, cand, kelly))
    return sorted(specs, key=lambda x: (x.min_confidence, x.min_ev, str(x.min_probability), str(x.min_odds), str(x.max_odds), x.max_candidates, x.fractional_kelly))


def run_search_stage(
    stage_name: str,
    specs: list[SearchSpec],
    races: dict[int, RaceCache],
    ticket_type: TicketType,
    logger: ProgressLogger,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    best: EvalResult | None = None
    valid_count = 0
    total = len(specs)
    label = TICKET_LABELS[ticket_type]
    for idx, spec in enumerate(specs, 1):
        result = evaluate_spec(races, spec)
        results.append(result)
        if passes_constraints(result):
            valid_count += 1
            if best is None or (result.profit, result.races_bet) > (best.profit, best.races_bet):
                best = result
        logger.maybe_log(
            phase=stage_name,
            ticket_label=label,
            done=idx,
            total=total,
            best=best,
            valid_candidates=valid_count,
        )
    return results


def cross_validate(candidate: EvalResult, caches: dict[int, RaceCache]) -> dict[str, Any]:
    positive_folds = 0
    folds: dict[str, float] = {}
    for name, (start, end) in FOLD_RANGES.items():
        races = {k: v for k, v in caches.items() if start <= v.race_date <= end}
        result = evaluate_spec(races, candidate.spec)
        folds[name] = result.roi
        if result.roi > 0:
            positive_folds += 1
    return {"positive_folds": positive_folds, "fold_rois": folds, "stable": positive_folds >= 2}


def holdout_validate(candidates: list[EvalResult], holdout_races: dict[int, RaceCache]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        result = evaluate_spec(holdout_races, candidate.spec)
        out.append({"dev": candidate, "holdout": result})
    return out


def combine_portfolio(selected: dict[TicketType, SearchSpec], holdout_races: dict[int, RaceCache]) -> dict[str, Any]:
    race_totals: dict[int, dict[str, Any]] = {}
    contrib: dict[str, float] = defaultdict(float)

    for tt, spec in selected.items():
        for rc in holdout_races.values():
            result = evaluate_spec({rc.race_id: rc}, spec)
            if result.total_stake <= 0:
                continue
            row = race_totals.setdefault(
                rc.race_id,
                {"race_date": rc.race_date, "stake": 0.0, "payout": 0.0, "ticket_types": set()},
            )
            row["stake"] += result.total_stake
            row["payout"] += result.total_payout
            row["ticket_types"].add(TICKET_LABELS[tt])
            contrib[TICKET_LABELS[tt]] += result.profit

    total_stake = sum(v["stake"] for v in race_totals.values())
    total_payout = sum(v["payout"] for v in race_totals.values())
    total_profit = total_payout - total_stake
    months = defaultdict(lambda: {"stake": 0.0, "payout": 0.0})
    overlap_count = 0
    for row in race_totals.values():
        month = row["race_date"].strftime("%Y-%m")
        months[month]["stake"] += row["stake"]
        months[month]["payout"] += row["payout"]
        if len(row["ticket_types"]) > 1:
            overlap_count += 1

    return {
        "total_stake": total_stake,
        "total_payout": total_payout,
        "total_profit": total_profit,
        "roi": total_profit / total_stake if total_stake > 0 else 0.0,
        "bet_races": len(race_totals),
        "overlap_races": overlap_count,
        "overlap_rate": overlap_count / len(race_totals) if race_totals else 0.0,
        "avg_stake_per_race": total_stake / len(race_totals) if race_totals else 0.0,
        "ticket_profit_contrib": dict(contrib),
        "monthly_roi": {
            m: ((v["payout"] - v["stake"]) / v["stake"]) if v["stake"] > 0 else 0.0
            for m, v in months.items()
        },
    }


def serialize_eval(result: EvalResult) -> dict[str, Any]:
    return {
        "ticket_type": TICKET_LABELS[result.spec.ticket_type],
        "params": asdict(result.spec),
        "params_readable": {
            "min_confidence": result.spec.min_confidence,
            "min_ev": result.spec.min_ev,
            "min_probability": result.spec.min_probability,
            "min_odds": result.spec.min_odds,
            "max_odds": result.spec.max_odds,
            "max_candidates": result.spec.max_candidates,
            "fractional_kelly": result.spec.fractional_kelly,
        },
        "profit": result.profit,
        "stake": result.total_stake,
        "payout": result.total_payout,
        "roi": result.roi,
        "races_bet": result.races_bet,
        "hit_races": result.hit_races,
        "hit_rate": result.hit_rate,
        "monthly_roi": result.monthly_roi,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", default="v010")
    args = parser.parse_args()

    logger = ProgressLogger()
    print("=" * 90)
    print(f"  回収額最大化探索 v4 (model={args.model_version})")
    print("=" * 90)

    caches = precompute(args.model_version)
    if not caches:
        raise SystemExit("データなし")

    dev_races = {k: v for k, v in caches.items() if DEV_START <= v.race_date <= DEV_END}
    holdout_races = {k: v for k, v in caches.items() if HOLDOUT_START <= v.race_date <= HOLDOUT_END}
    print(f"  開発: {len(dev_races)}R | ホールドアウト: {len(holdout_races)}R", flush=True)

    summary: dict[str, Any] = {
        "model_version": args.model_version,
        "dev_period": [DEV_START.isoformat(), DEV_END.isoformat()],
        "holdout_period": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "tickets": {},
    }

    selected_specs: dict[TicketType, SearchSpec] = {}

    for ticket_type in TICKET_TYPES:
        print("\n" + "=" * 90)
        print(f"  券種: {TICKET_LABELS[ticket_type]}")
        print("=" * 90)
        coarse_specs = coarse_specs_for_ticket(ticket_type)
        coarse_results = run_search_stage("coarse", coarse_specs, dev_races, ticket_type, logger)
        valid_coarse = [r for r in coarse_results if passes_constraints(r)]
        coarse_front = pareto_front(valid_coarse)
        coarse_front = coarse_front[:TOP_SEEDS]

        fine_specs = fine_specs_from_seeds(ticket_type, coarse_front)
        fine_results = run_search_stage("fine", fine_specs, dev_races, ticket_type, logger)
        valid_fine = [r for r in fine_results if passes_constraints(r)]
        fine_front = pareto_front(valid_fine)
        fine_front = fine_front[:TOP_FINE]

        cv_rows = []
        for candidate in fine_front:
            cv = cross_validate(candidate, dev_races)
            cv_rows.append({"candidate": candidate, **cv})
        stable_rows = [row for row in cv_rows if row["stable"]]

        holdout_rows = holdout_validate([row["candidate"] for row in stable_rows], holdout_races)
        holdout_rows.sort(
            key=lambda row: (row["holdout"].profit, row["holdout"].races_bet, row["holdout"].roi),
            reverse=True,
        )

        best_row = None
        for row in holdout_rows:
            if passes_constraints(row["holdout"]):
                best_row = row
                selected_specs[ticket_type] = row["holdout"].spec
                break

        summary["tickets"][TICKET_LABELS[ticket_type]] = {
            "coarse_top": [serialize_eval(r) for r in coarse_front],
            "fine_top": [serialize_eval(r) for r in fine_front],
            "cv": [
                {
                    "candidate": serialize_eval(row["candidate"]),
                    "positive_folds": row["positive_folds"],
                    "fold_rois": row["fold_rois"],
                    "stable": row["stable"],
                }
                for row in stable_rows[:10]
            ],
            "holdout": [
                {
                    "dev": serialize_eval(row["dev"]),
                    "holdout": serialize_eval(row["holdout"]),
                }
                for row in holdout_rows[:10]
            ],
            "selected": {
                "dev": serialize_eval(best_row["dev"]),
                "holdout": serialize_eval(best_row["holdout"]),
            } if best_row else None,
        }

        if best_row:
            print(
                f"  採用候補: ROI={best_row['holdout'].roi*100:+.1f}% "
                f"profit={best_row['holdout'].profit:+,.0f}円 bet_races={best_row['holdout'].races_bet}",
                flush=True,
            )
        else:
            print("  採用候補: なし", flush=True)

    print("\n" + "=" * 90)
    print("  ポートフォリオ検証")
    print("=" * 90)
    portfolio_candidates = [
        [WIN, WIDE],
        [WIN, QUINELLA],
        [WIN, WIDE, QUINELLA],
        [WIN, WIDE, QUINELLA, EXACTA],
    ]
    portfolio_results = []
    for combo in portfolio_candidates:
        if not all(tt in selected_specs for tt in combo):
            continue
        result = combine_portfolio({tt: selected_specs[tt] for tt in combo}, holdout_races)
        portfolio_results.append(
            {
                "tickets": [TICKET_LABELS[tt] for tt in combo],
                **result,
            }
        )
        print(
            f"  {' + '.join(TICKET_LABELS[tt] for tt in combo)}: "
            f"ROI={result['roi']*100:+.1f}% profit={result['total_profit']:+,.0f}円 "
            f"bet_races={result['bet_races']} overlap={result['overlap_rate']*100:.1f}%",
            flush=True,
        )

    summary["portfolio"] = portfolio_results

    out_dir = Path("data/strategy_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{args.model_version}_bet_optimization_v4.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  結果保存: {out_path}", flush=True)


if __name__ == "__main__":
    main()
