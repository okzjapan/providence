"""Rendering helpers for backtest output."""

from __future__ import annotations

from rich.table import Table

from providence.backtest.types import BacktestSummary


def summary_table(summary: BacktestSummary) -> Table:
    table = Table(title="Backtest Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total races", str(summary.total_races))
    table.add_row("Profit evaluated", str(summary.profit_evaluated_races))
    table.add_row("Accuracy only", str(summary.accuracy_only_races))
    table.add_row("Total stake", f"{summary.total_stake:.0f}")
    table.add_row("Total payout", f"{summary.total_payout:.0f}")
    table.add_row("Total profit", f"{summary.total_profit:.0f}")
    table.add_row("ROI", f"{summary.roi:.3f}")
    table.add_row("Hit rate", f"{summary.hit_rate:.3f}")
    table.add_row("Max drawdown", f"{summary.max_drawdown:.0f}")
    table.add_row("Sharpe", f"{summary.sharpe_ratio:.3f}")
    return table
