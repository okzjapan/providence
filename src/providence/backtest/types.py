"""Backtest dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from providence.strategy.types import RecommendedBet, StrategyRunResult


@dataclass(frozen=True)
class SettledRecommendation:
    recommendation: RecommendedBet
    payout_amount: float
    profit: float
    hit: bool


@dataclass
class BacktestRaceResult:
    race_id: int
    race_date: date
    race_number: int
    track_id: int
    judgment_time: datetime
    strategy_result: StrategyRunResult
    settled_recommendations: list[SettledRecommendation] = field(default_factory=list)
    profit_evaluated: bool = False
    total_profit: float = 0.0
    total_stake: float = 0.0
    total_payout: float = 0.0


@dataclass(frozen=True)
class BacktestSummary:
    total_races: int
    profit_evaluated_races: int
    accuracy_only_races: int
    total_stake: float
    total_payout: float
    total_profit: float
    roi: float
    hit_rate: float
    max_drawdown: float
    sharpe_ratio: float
