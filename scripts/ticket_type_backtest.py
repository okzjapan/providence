"""Per-ticket-type backtest analysis.

Runs model prediction once and evaluates strategy performance
for each ticket type independently with multiple parameter sets.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time

import polars as pl

from providence.backtest.settlement import settle_recommendations
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TicketType
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.strategy.normalize import (
    flatten_ticket_probs,
    market_odds_from_rows,
    payouts_from_rows,
)
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    RacePredictionBundle,
    MarketTicketOdds,
    SettledTicketPayout,
    StrategyConfig,
)

START_DATE = date(2025, 12, 1)
END_DATE = date(2026, 3, 31)
JUDGMENT_CLOCK = dt_time(15, 0, 0)

TICKET_TYPES = [
    TicketType.WIN,
    TicketType.EXACTA,
    TicketType.QUINELLA,
    TicketType.WIDE,
    TicketType.TRIFECTA,
    TicketType.TRIO,
]

CONFIGS: dict[str, StrategyConfig] = {
    "default": StrategyConfig(
        fractional_kelly=0.25,
        max_candidates=12,
        max_total_stake=10_000,
        min_confidence=0.1,
    ),
    "permissive": StrategyConfig(
        fractional_kelly=0.5,
        max_candidates=30,
        max_total_stake=50_000,
        min_confidence=0.05,
        min_expected_value=-0.1,
    ),
}


@dataclass
class RaceContext:
    bundle: RacePredictionBundle
    market_odds: list[MarketTicketOdds]
    payouts: list[SettledTicketPayout]
    race_date: date
    judgment_time: datetime


@dataclass
class TicketMetrics:
    total_races_bet: int = 0
    total_bets: int = 0
    total_stake: float = 0.0
    total_payout: float = 0.0
    total_profit: float = 0.0
    hit_races: int = 0
    hit_bets: int = 0
    avg_odds_on_hit: float = 0.0
    max_single_payout: float = 0.0
    _hit_odds: list[float] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def hit_rate_by_race(self) -> float:
        return self.hit_races / self.total_races_bet if self.total_races_bet > 0 else 0.0

    @property
    def hit_rate_by_bet(self) -> float:
        return self.hit_bets / self.total_bets if self.total_bets > 0 else 0.0

    @property
    def avg_stake_per_race(self) -> float:
        return self.total_stake / self.total_races_bet if self.total_races_bet > 0 else 0.0

    def finalize(self) -> None:
        if self._hit_odds:
            self.avg_odds_on_hit = sum(self._hit_odds) / len(self._hit_odds)


def load_predictions_and_data(
    start_date: date,
    end_date: date,
    judgment_clock: dt_time,
) -> dict[int, RaceContext]:
    print(f"[1/3] 特徴量をロード中...", flush=True)
    loader = DataLoader()
    pipeline = FeaturePipeline()
    model_store = ModelStore()
    repository = Repository()
    session_factory = get_session_factory()

    raw_df = loader.load_race_dataset(end_date=end_date)
    if raw_df.is_empty():
        print("  データが見つかりません")
        return {}

    from pathlib import Path

    cache_key = FeaturePipeline.cache_key(
        {
            "purpose": "backtest",
            "rows": len(raw_df),
            "race_min": raw_df["race_id"].min(),
            "race_max": raw_df["race_id"].max(),
            "entry_max": raw_df["race_entry_id"].max(),
            "date_min": raw_df["race_date"].min(),
            "date_max": raw_df["race_date"].max(),
            "end_date": end_date.isoformat(),
        }
    )
    cache_path = Path("data/processed") / f"backtest_features_{cache_key}.parquet"
    feature_dataset = pipeline.build_and_cache(raw_df, str(cache_path))

    dataset = feature_dataset.filter(
        pl.col("race_date").is_between(start_date, end_date, closed="both")
    )
    if dataset.is_empty():
        print("  期間内データなし")
        return {}

    race_rows = (
        dataset.select(["race_id", "race_date", "track_id", "race_number"])
        .unique(maintain_order=True)
        .sort(["race_date", "race_number", "race_id"])
    )
    unique_dates = race_rows.select("race_date").unique(maintain_order=True)
    race_dates = [row["race_date"] for row in unique_dates.iter_rows(named=True)]

    total_races = race_rows.height
    print(f"  {len(race_dates)}日間, {total_races}レース")

    print(f"[2/3] モデル予測を実行中...", flush=True)
    predictor = Predictor(model_store, pipeline, loader, version="latest")
    race_data: dict[int, RaceContext] = {}
    processed = 0

    for race_date in race_dates:
        day_features = dataset.filter(pl.col("race_date") == race_date).sort(
            ["race_number", "race_id", "post_position"]
        )
        bundles = predictor.predict_feature_races(day_features)
        race_ids = list(bundles)
        judgment_time = datetime.combine(race_date, judgment_clock)

        with session_factory() as session:
            market_rows_by_race = repository.get_latest_market_odds_for_races(
                session, race_ids, judgment_time=judgment_time,
            )
            payout_rows_by_race = repository.get_ticket_payouts_for_races(session, race_ids)

            for race_id, bundle in bundles.items():
                market_odds = market_odds_from_rows(
                    market_rows_by_race.get(race_id, [])
                )
                payout_rows = payout_rows_by_race.get(race_id, [])
                payouts = payouts_from_rows(payout_rows)
                race_data[race_id] = RaceContext(
                    bundle=bundle,
                    market_odds=market_odds,
                    payouts=payouts,
                    race_date=race_date,
                    judgment_time=judgment_time,
                )
                processed += 1

        print(f"\r  予測完了: {processed}/{total_races} レース", end="", flush=True)

    print()
    return race_data


def run_ticket_type_analysis(
    race_data: dict[int, RaceContext],
    config_name: str,
    base_config: StrategyConfig,
) -> dict[TicketType, TicketMetrics]:
    results: dict[TicketType, TicketMetrics] = {tt: TicketMetrics() for tt in TICKET_TYPES}

    for ticket_type in TICKET_TYPES:
        config = StrategyConfig(
            fractional_kelly=base_config.fractional_kelly,
            min_bet_amount=base_config.min_bet_amount,
            max_total_stake=base_config.max_total_stake,
            min_weight_threshold=base_config.min_weight_threshold,
            min_expected_value=base_config.min_expected_value,
            min_probability=base_config.min_probability,
            max_candidates=base_config.max_candidates,
            min_confidence=base_config.min_confidence,
            allowed_ticket_types=frozenset({ticket_type}),
        )
        metrics = results[ticket_type]

        for race_id, ctx in race_data.items():
            if not ctx.market_odds or not ctx.payouts:
                continue

            decision_context = DecisionContext(
                judgment_time=ctx.judgment_time,
                evaluation_mode=EvaluationMode.FIXED,
                provenance="backtest-analysis",
            )

            strategy_result = run_strategy(
                ctx.bundle,
                ctx.market_odds,
                decision_context=decision_context,
                config=config,
            )

            if not strategy_result.recommended_bets:
                continue

            settled = settle_recommendations(strategy_result.recommended_bets, ctx.payouts)
            race_stake = sum(item.recommendation.recommended_bet for item in settled)
            race_payout = sum(item.payout_amount for item in settled)
            race_hit = any(item.hit for item in settled)

            metrics.total_races_bet += 1
            metrics.total_bets += len(settled)
            metrics.total_stake += race_stake
            metrics.total_payout += race_payout
            metrics.total_profit += race_payout - race_stake

            if race_hit:
                metrics.hit_races += 1
            for item in settled:
                if item.hit:
                    metrics.hit_bets += 1
                    metrics._hit_odds.append(item.recommendation.odds_value)
                    if item.payout_amount > metrics.max_single_payout:
                        metrics.max_single_payout = item.payout_amount

        metrics.finalize()

    return results


def run_combined_analysis(
    race_data: dict[int, RaceContext],
    config_name: str,
    base_config: StrategyConfig,
) -> dict[str, TicketMetrics]:
    """Run with all ticket types enabled and break down results by ticket type."""
    config = StrategyConfig(
        fractional_kelly=base_config.fractional_kelly,
        min_bet_amount=base_config.min_bet_amount,
        max_total_stake=base_config.max_total_stake,
        min_weight_threshold=base_config.min_weight_threshold,
        min_expected_value=base_config.min_expected_value,
        min_probability=base_config.min_probability,
        max_candidates=base_config.max_candidates,
        min_confidence=base_config.min_confidence,
        allowed_ticket_types=None,
    )

    per_type: dict[TicketType, TicketMetrics] = {tt: TicketMetrics() for tt in TICKET_TYPES}
    overall = TicketMetrics()

    for race_id, ctx in race_data.items():
        if not ctx.market_odds or not ctx.payouts:
            continue

        decision_context = DecisionContext(
            judgment_time=ctx.judgment_time,
            evaluation_mode=EvaluationMode.FIXED,
            provenance="backtest-analysis",
        )

        strategy_result = run_strategy(
            ctx.bundle,
            ctx.market_odds,
            decision_context=decision_context,
            config=config,
        )

        if not strategy_result.recommended_bets:
            continue

        settled = settle_recommendations(strategy_result.recommended_bets, ctx.payouts)
        race_stake = sum(item.recommendation.recommended_bet for item in settled)
        race_payout = sum(item.payout_amount for item in settled)

        overall.total_races_bet += 1
        overall.total_bets += len(settled)
        overall.total_stake += race_stake
        overall.total_payout += race_payout
        overall.total_profit += race_payout - race_stake
        if any(item.hit for item in settled):
            overall.hit_races += 1

        by_type_this_race: dict[TicketType, list] = defaultdict(list)
        for item in settled:
            by_type_this_race[item.recommendation.ticket_type].append(item)

        for tt, items in by_type_this_race.items():
            if tt not in per_type:
                continue
            m = per_type[tt]
            type_stake = sum(i.recommendation.recommended_bet for i in items)
            type_payout = sum(i.payout_amount for i in items)
            m.total_races_bet += 1
            m.total_bets += len(items)
            m.total_stake += type_stake
            m.total_payout += type_payout
            m.total_profit += type_payout - type_stake
            if any(i.hit for i in items):
                m.hit_races += 1
            for i in items:
                if i.hit:
                    m.hit_bets += 1
                    m._hit_odds.append(i.recommendation.odds_value)
                    if i.payout_amount > m.max_single_payout:
                        m.max_single_payout = i.payout_amount

        for i in settled:
            if i.hit:
                overall.hit_bets += 1
                overall._hit_odds.append(i.recommendation.odds_value)
                if i.payout_amount > overall.max_single_payout:
                    overall.max_single_payout = i.payout_amount

    overall.finalize()
    for m in per_type.values():
        m.finalize()

    combined: dict[str, TicketMetrics] = {"総合": overall}
    for tt in TICKET_TYPES:
        combined[tt.value] = per_type[tt]
    return combined


def print_results(
    independent: dict[str, dict[TicketType, TicketMetrics]],
    combined: dict[str, dict[str, TicketMetrics]],
    total_eligible_races: int,
) -> None:
    TT_LABELS = {
        TicketType.WIN: "単勝",
        TicketType.EXACTA: "2連単",
        TicketType.QUINELLA: "2連複",
        TicketType.WIDE: "ワイド",
        TicketType.TRIFECTA: "3連単",
        TicketType.TRIO: "3連複",
    }

    print("\n" + "=" * 100)
    print(f"  バックテスト結果  |  期間: {START_DATE} 〜 {END_DATE}  |  評価対象: {total_eligible_races} レース")
    print("=" * 100)

    for config_name in independent:
        config_label = "デフォルト設定" if config_name == "default" else "パーミッシブ設定"
        config_desc = CONFIGS[config_name]
        print(f"\n{'─' * 100}")
        print(f"  ■ {config_label}  (kelly={config_desc.fractional_kelly}, candidates={config_desc.max_candidates}, max_stake={config_desc.max_total_stake:,}円/R)")
        print(f"{'─' * 100}")

        print(f"\n  【A. 券種別独立バックテスト】（各券種を単独で運用した場合）\n")
        header = f"  {'券種':<8} {'投票R数':>8} {'投票数':>8} {'総投下額':>12} {'総回収額':>12} {'総損益':>12} {'回収率':>8} {'的中率(R)':>10} {'的中率(票)':>10} {'平均投下/R':>10} {'的中時平均倍率':>14} {'最大払戻':>10}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for tt in TICKET_TYPES:
            m = independent[config_name][tt]
            label = TT_LABELS[tt]
            if m.total_races_bet == 0:
                print(f"  {label:<8} {'N/A':>8} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>14} {'N/A':>10}")
                continue
            roi_pct = f"{(1 + m.roi) * 100:.1f}%"
            hr_r = f"{m.hit_rate_by_race * 100:.1f}%"
            hr_b = f"{m.hit_rate_by_bet * 100:.1f}%"
            avg_odds = f"{m.avg_odds_on_hit:.1f}倍" if m.avg_odds_on_hit > 0 else "N/A"
            print(
                f"  {label:<8} {m.total_races_bet:>8,} {m.total_bets:>8,} "
                f"{m.total_stake:>12,.0f} {m.total_payout:>12,.0f} "
                f"{m.total_profit:>+12,.0f} {roi_pct:>8} "
                f"{hr_r:>10} {hr_b:>10} "
                f"{m.avg_stake_per_race:>10,.0f} {avg_odds:>14} "
                f"{m.max_single_payout:>10,.0f}"
            )

        print(f"\n  【B. 全券種混合バックテスト】（全券種を同時に運用した場合の内訳）\n")
        header2 = f"  {'券種':<8} {'投票R数':>8} {'投票数':>8} {'総投下額':>12} {'総回収額':>12} {'総損益':>12} {'回収率':>8} {'的中率(R)':>10} {'的中率(票)':>10}"
        print(header2)
        print("  " + "─" * (len(header2) - 2))

        for label_key, m in combined[config_name].items():
            is_total = label_key == "総合"
            label = f"{'★ ' if is_total else ''}{label_key}"
            if m.total_races_bet == 0:
                print(f"  {label:<8} {'N/A':>8} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
                continue
            roi_pct = f"{(1 + m.roi) * 100:.1f}%"
            hr_r = f"{m.hit_rate_by_race * 100:.1f}%"
            hr_b = f"{m.hit_rate_by_bet * 100:.1f}%"
            prefix = "  " if not is_total else "  "
            print(
                f"{prefix}{label:<8} {m.total_races_bet:>8,} {m.total_bets:>8,} "
                f"{m.total_stake:>12,.0f} {m.total_payout:>12,.0f} "
                f"{m.total_profit:>+12,.0f} {roi_pct:>8} "
                f"{hr_r:>10} {hr_b:>10}"
            )

    print(f"\n{'=' * 100}")
    print("  注: 回収率 = 総回収額 / 総投下額 × 100%。100%超で黒字。")
    print("  注: 的中率(R) = 1件以上当たったレース / 投票したレース。")
    print("  注: 的中率(票) = 的中した投票数 / 全投票数。")
    print(f"{'=' * 100}\n")


def main() -> None:
    t0 = time.time()

    race_data = load_predictions_and_data(START_DATE, END_DATE, JUDGMENT_CLOCK)
    if not race_data:
        print("データが見つかりません。終了します。")
        sys.exit(1)

    total_eligible = sum(
        1 for ctx in race_data.values() if ctx.market_odds and ctx.payouts
    )
    print(f"  オッズ+払戻両方あり: {total_eligible}/{len(race_data)} レース")

    independent_results: dict[str, dict[TicketType, TicketMetrics]] = {}
    combined_results: dict[str, dict[str, TicketMetrics]] = {}

    print(f"\n[3/3] 戦略バックテスト実行中...", flush=True)
    for config_name, config in CONFIGS.items():
        config_label = "デフォルト" if config_name == "default" else "パーミッシブ"
        print(f"  {config_label}設定: 券種別独立分析...", flush=True)
        independent_results[config_name] = run_ticket_type_analysis(
            race_data, config_name, config,
        )
        print(f"  {config_label}設定: 全券種混合分析...", flush=True)
        combined_results[config_name] = run_combined_analysis(
            race_data, config_name, config,
        )

    print_results(independent_results, combined_results, total_eligible)

    elapsed = time.time() - t0
    print(f"実行時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
