"""Prediction CLI command."""

from __future__ import annotations

from datetime import date

import typer
from rich.console import Console
from rich.table import Table

from providence.cli.strategy_options import build_strategy_config
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.services.prediction_runner import (
    PredictionResult,
    resolve_judgment_time,
    run_prediction,
    save_prediction,
)
from providence.strategy.normalize import format_combination

console = Console()


def predict_command(
    date_str: str = typer.Option(..., "--date", help="Race date (YYYY-MM-DD)"),
    track: str = typer.Option(..., "--track", help="Track name"),
    race: int = typer.Option(..., "--race", help="Race number"),
    model_version: str = typer.Option("latest", "--model-version", help="Model version to use"),
    judgment_time: str | None = typer.Option(None, "--judgment-time", help="ISO8601 cutoff time for odds selection"),
    save: bool = typer.Option(False, "--save", help="Persist the strategy run"),
    ticket_types: str | None = typer.Option(None, "--ticket-types", help="Comma-separated ticket types to bet on (e.g. win,wide)"),
    max_candidates: int | None = typer.Option(None, "--max-candidates", help="Max candidate bets per race"),
    fractional_kelly: float | None = typer.Option(None, "--fractional-kelly", help="Kelly fraction multiplier (default 0.25)"),
    min_confidence: float | None = typer.Option(None, "--min-confidence", help="Minimum race confidence score (default 0.1)"),
    min_expected_value: float | None = typer.Option(None, "--min-expected-value", help="Minimum expected value threshold (default 0.0)"),
) -> None:
    target_date = date.fromisoformat(date_str)
    track_code = TrackCode.from_name(track)
    decision_time = resolve_judgment_time(judgment_time)
    session_factory = get_session_factory()
    repo = Repository()
    loader = DataLoader()
    predictor = Predictor(ModelStore(), FeaturePipeline(), loader, version=model_version)
    config = build_strategy_config(
        ticket_types=ticket_types,
        max_candidates=max_candidates,
        fractional_kelly=fractional_kelly,
        min_confidence=min_confidence,
        min_expected_value=min_expected_value,
    )

    try:
        result = run_prediction(
            target_date=target_date,
            track_code=track_code,
            race_number=race,
            decision_time=decision_time,
            session_factory=session_factory,
            repo=repo,
            predictor=predictor,
            loader=loader,
            config=config,
            provenance="cli.predict",
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    _display_warnings(result)
    _display_confidence(result)
    _display_probabilities(result)
    _display_recommendations(result)

    if save:
        save_prediction(result, decision_time, session_factory, repo)
        console.print("[green]Strategy run saved.[/green]")


def _display_warnings(result: PredictionResult) -> None:
    if result.program_sync_warning is not None:
        console.print(f"[yellow]{result.program_sync_warning}[/yellow]")
    if result.missing_trial_positions and result.program_sync_warning is None:
        cars = ", ".join(str(p) for p in result.missing_trial_positions)
        console.print(
            "[yellow]Warning: 最新 Program 同期後も試走タイムが未取得の車があります。"
            f" 対象車番: {cars}. 予測精度に影響する可能性があります。[/yellow]"
        )
    if result.strategy.skip_reason:
        console.print(f"[yellow]Skip reason: {result.strategy.skip_reason}[/yellow]")
    elif not result.has_market_odds:
        console.print("[yellow]No eligible market odds found before judgment time.[/yellow]")


def _display_confidence(result: PredictionResult) -> None:
    score = result.strategy.confidence_score
    console.print(f"Confidence: {score:.4f}")


def _display_probabilities(result: PredictionResult) -> None:
    bundle = result.bundle
    table = Table(title=f"{result.track_code.japanese_name} {result.target_date} R{result.race_number}")
    table.add_column("Car", justify="right")
    table.add_column("WinProb", justify="right")
    for index, prob in sorted(bundle.ticket_probs["win"].items()):
        car = bundle.index_map.post_position_for_index(index)
        table.add_row(str(car), f"{prob:.3f}")
    console.print(table)


def _display_recommendations(result: PredictionResult) -> None:
    strategy = result.strategy
    table = Table(title="Strategy Recommendations")
    table.add_column("Ticket")
    table.add_column("Combination", justify="right")
    table.add_column("Prob", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Bet", justify="right")

    display_bets = strategy.recommended_bets or strategy.candidate_bets
    if display_bets:
        selected_keys = {
            (b.ticket_type, b.combination): b for b in strategy.recommended_bets
        }
        for bet in display_bets:
            matched = selected_keys.get((bet.ticket_type, bet.combination), bet)
            final_bet = matched.recommended_bet if strategy.recommended_bets else 0.0
            table.add_row(
                bet.ticket_type.value,
                format_combination(bet.ticket_type, bet.combination),
                f"{bet.probability:.3f}",
                f"{bet.odds_value:.2f}",
                f"{bet.expected_value:.3f}",
                f"{bet.stake_weight:.4f}",
                f"{final_bet:.0f}",
            )
    else:
        table.add_row("-", "-", "-", "-", "-", "-", "-")
    console.print(table)
