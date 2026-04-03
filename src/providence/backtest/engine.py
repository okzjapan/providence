"""Fixed and walk-forward backtest engine."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.strategy.normalize import market_odds_from_rows, payouts_from_rows
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import DecisionContext, EvaluationMode, StrategyConfig

from .settlement import settle_recommendations
from .types import BacktestRaceResult


class BacktestEngine:
    def __init__(
        self,
        *,
        loader: DataLoader | None = None,
        pipeline: FeaturePipeline | None = None,
        model_store: ModelStore | None = None,
        repository: Repository | None = None,
    ) -> None:
        self.loader = loader or DataLoader()
        self.pipeline = pipeline or FeaturePipeline()
        self.model_store = model_store or ModelStore()
        self.repository = repository or Repository()
        self.session_factory = get_session_factory()

    def run(
        self,
        *,
        start_date: date,
        end_date: date,
        judgment_clock: time,
        bankroll: float,
        evaluation_mode: EvaluationMode,
        model_version: str = "latest",
        track_id: int | None = None,
        config: StrategyConfig | None = None,
    ) -> list[BacktestRaceResult]:
        config = config or StrategyConfig()
        feature_dataset = self._load_feature_dataset(end_date=end_date)
        if feature_dataset.is_empty():
            return []
        dataset = feature_dataset.filter(
            pl.col("race_date").is_between(start_date, end_date, closed="both")
        )
        if track_id is not None and not dataset.is_empty():
            dataset = dataset.filter(pl.col("track_id") == track_id)
        if dataset.is_empty():
            return []

        race_rows = (
            dataset.select(["race_id", "race_date", "track_id", "race_number"])
            .unique(maintain_order=True)
            .sort(["race_date", "race_number", "race_id"])
        )
        race_meta = {int(row["race_id"]): row for row in race_rows.iter_rows(named=True)}
        race_dates = [row["race_date"] for row in race_rows.select("race_date").unique(maintain_order=True).iter_rows(named=True)]

        predictors: dict[str, Predictor] = {}
        results: list[BacktestRaceResult] = []
        current_bankroll = bankroll

        for race_date in race_dates:
            version = model_version
            if evaluation_mode == EvaluationMode.WALK_FORWARD:
                try:
                    _, metadata = self.model_store.load_for_backtest(race_date, mode="walk-forward")
                except FileNotFoundError:
                    continue
                version = str(metadata["version"])
            predictor = predictors.get(version)
            if predictor is None:
                predictor = Predictor(self.model_store, self.pipeline, self.loader, version=version)
                predictors[version] = predictor
            day_features = dataset.filter(pl.col("race_date") == race_date).sort(["race_number", "race_id", "post_position"])
            bundles = predictor.predict_feature_races(day_features)
            race_ids = list(bundles)
            judgment_time = datetime.combine(race_date, judgment_clock)

            with self.session_factory() as session:
                market_rows_by_race = self.repository.get_latest_market_odds_for_races(
                    session,
                    race_ids,
                    judgment_time=judgment_time,
                )
                payout_rows_by_race = self.repository.get_ticket_payouts_for_races(session, race_ids)
                for race_id, bundle in bundles.items():
                    row = race_meta[race_id]
                    decision_context = DecisionContext(
                        judgment_time=judgment_time,
                        evaluation_mode=evaluation_mode,
                        timezone="UTC",
                        provenance="backtest",
                    )

                    market_odds = market_odds_from_rows(market_rows_by_race.get(bundle.race_id, []))
                    payouts = payouts_from_rows(payout_rows_by_race.get(bundle.race_id, []))
                    strategy_result = run_strategy(
                        bundle,
                        market_odds,
                        decision_context=decision_context,
                        bankroll=current_bankroll,
                        config=config,
                    )
                    settled = settle_recommendations(strategy_result.recommended_bets, payouts) if payouts else []
                    total_stake = float(sum(item.recommendation.recommended_bet for item in settled))
                    total_payout = float(sum(item.payout_amount for item in settled))
                    total_profit = float(sum(item.profit for item in settled))
                    profit_evaluated = bool(market_odds and payouts)
                    if profit_evaluated:
                        current_bankroll += total_profit

                    results.append(
                        BacktestRaceResult(
                            race_id=bundle.race_id,
                            race_date=race_date,
                            race_number=int(row["race_number"]),
                            track_id=int(row["track_id"]),
                            judgment_time=judgment_time,
                            strategy_result=strategy_result,
                            settled_recommendations=settled,
                            profit_evaluated=profit_evaluated,
                            total_profit=total_profit,
                            total_stake=total_stake,
                            total_payout=total_payout,
                        )
                    )

        return results

    def _load_feature_dataset(self, *, end_date: date) -> pl.DataFrame:
        raw_df = self.loader.load_race_dataset(end_date=end_date)
        if raw_df.is_empty():
            return raw_df

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
        return self.pipeline.build_and_cache(raw_df, str(cache_path))
