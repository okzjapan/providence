"""Backtest CLI command."""

from __future__ import annotations

from datetime import date, datetime

import typer
from rich.console import Console

from providence.backtest.engine import BacktestEngine
from providence.backtest.metrics import summarize_backtest
from providence.backtest.report import summary_table
from providence.config import DEFAULT_BANKROLL_JPY
from providence.domain.enums import TrackCode
from providence.strategy.types import EvaluationMode

console = Console()


def backtest_command(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    judgment_time: str = typer.Option(..., "--judgment-time", help="Daily cutoff time (HH:MM or HH:MM:SS)"),
    bankroll: float = typer.Option(DEFAULT_BANKROLL_JPY, "--bankroll", help="Initial bankroll"),
    model_version: str = typer.Option("latest", "--model-version", help="Model version"),
    evaluation_mode: str = typer.Option("fixed", "--evaluation-mode", help="fixed or walk-forward"),
    track: str | None = typer.Option(None, "--track", help="Track name"),
) -> None:
    mode = EvaluationMode.FIXED if evaluation_mode == "fixed" else EvaluationMode.WALK_FORWARD
    track_code = TrackCode.from_name(track) if track else None
    engine = BacktestEngine()
    results = engine.run(
        start_date=date.fromisoformat(from_date),
        end_date=date.fromisoformat(to_date),
        judgment_clock=_parse_clock(judgment_time),
        bankroll=bankroll,
        evaluation_mode=mode,
        model_version=model_version,
        track_id=track_code.value if track_code else None,
    )
    summary = summarize_backtest(results)
    console.print(summary_table(summary))


def _parse_clock(value: str):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    raise typer.BadParameter("judgment-time must be HH:MM or HH:MM:SS")
