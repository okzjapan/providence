"""Aggregate metrics for backtests."""

from __future__ import annotations

import math
from collections import defaultdict

from .types import BacktestRaceResult, BacktestSummary


def summarize_backtest(results: list[BacktestRaceResult]) -> BacktestSummary:
    total_races = len(results)
    profit_results = [row for row in results if row.profit_evaluated]
    accuracy_only_races = total_races - len(profit_results)
    total_stake = float(sum(row.total_stake for row in profit_results))
    total_payout = float(sum(row.total_payout for row in profit_results))
    total_profit = float(sum(row.total_profit for row in profit_results))
    roi = float(total_profit / total_stake) if total_stake > 0 else 0.0
    hit_count = sum(1 for row in profit_results if any(item.hit for item in row.settled_recommendations))
    hit_rate = float(hit_count / len(profit_results)) if profit_results else 0.0
    max_drawdown = _max_drawdown([row.total_profit for row in profit_results])
    sharpe_ratio = _daily_sharpe(results)
    return BacktestSummary(
        total_races=total_races,
        profit_evaluated_races=len(profit_results),
        accuracy_only_races=accuracy_only_races,
        total_stake=total_stake,
        total_payout=total_payout,
        total_profit=total_profit,
        roi=roi,
        hit_rate=hit_rate,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
    )


def _max_drawdown(profits: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return float(abs(max_drawdown))


def _daily_sharpe(results: list[BacktestRaceResult]) -> float:
    profits_by_day: dict[object, float] = defaultdict(float)
    for row in results:
        if row.profit_evaluated:
            profits_by_day[row.race_date] += row.total_profit
    if len(profits_by_day) < 2:
        return 0.0
    values = list(profits_by_day.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        return 0.0
    return float(mean / stdev)
