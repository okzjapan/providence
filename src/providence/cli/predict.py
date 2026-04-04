"""Prediction CLI command."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from providence.cli.strategy_options import build_strategy_config
from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.database.tables import Prediction, StrategyRun
from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.scraper.autorace_jp import AutoraceJpScraper
from providence.strategy.normalize import format_combination, market_odds_from_rows
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import DecisionContext, EvaluationMode, StrategyRunResult

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
) -> None:
    target_date = date.fromisoformat(date_str)
    track_code = TrackCode.from_name(track)
    decision_time = _resolve_judgment_time(judgment_time)
    loader = DataLoader()
    pipeline = FeaturePipeline()
    predictor = Predictor(ModelStore(), pipeline, loader, version=model_version)
    session_factory = get_session_factory()
    repo = Repository()
    program_sync_warning = asyncio.run(_refresh_race_entries(target_date, track_code, race, repo, session_factory))

    race_df = loader.load_race_dataset(start_date=target_date, end_date=target_date).filter(
        (pl.col("track_id") == track_code.value) & (pl.col("race_number") == race)
    )
    if race_df.is_empty():
        console.print("[red]対象レースのデータがありません。先に `providence scrape day` を実行してください。[/red]")
        raise typer.Exit(1)

    if program_sync_warning is not None:
        console.print(f"[yellow]{program_sync_warning}[/yellow]")
    missing_trial_positions = get_missing_trial_positions(race_df)
    if missing_trial_positions and program_sync_warning is None:
        cars = ", ".join(str(position) for position in missing_trial_positions)
        console.print(
            "[yellow]Warning: 最新 Program 同期後も試走タイムが未取得の車があります。"
            f" 対象車番: {cars}. 予測精度に影響する可能性があります。[/yellow]"
        )

    predictor.load_history(target_date)
    bundle = predictor.predict_race(race_df)

    with session_factory() as session:
        odds_rows = repo.get_latest_market_odds(session, bundle.race_id, judgment_time=decision_time)
    config = build_strategy_config(
        ticket_types=ticket_types,
        max_candidates=max_candidates,
        fractional_kelly=fractional_kelly,
    )
    market_odds = market_odds_from_rows(odds_rows)
    strategy = run_strategy(
        bundle,
        market_odds,
        decision_context=DecisionContext(
            judgment_time=decision_time,
            evaluation_mode=EvaluationMode.LIVE,
            timezone="UTC",
            provenance="cli.predict",
        ),
        config=config,
    )

    table = Table(title=f"{track_code.japanese_name} {target_date} R{race}")
    table.add_column("Car", justify="right")
    table.add_column("WinProb", justify="right")
    for index, prob in sorted(bundle.ticket_probs["win"].items()):
        car = bundle.index_map.post_position_for_index(index)
        table.add_row(str(car), f"{prob:.3f}")
    console.print(table)

    recommendation_table = Table(title="Strategy Recommendations")
    recommendation_table.add_column("Ticket")
    recommendation_table.add_column("Combination", justify="right")
    recommendation_table.add_column("Prob", justify="right")
    recommendation_table.add_column("Odds", justify="right")
    recommendation_table.add_column("EV", justify="right")
    recommendation_table.add_column("Weight", justify="right")
    recommendation_table.add_column("Bet", justify="right")

    display_bets = strategy.recommended_bets or strategy.candidate_bets
    if display_bets:
        selected_keys = {
            (b.ticket_type, b.combination): b for b in strategy.recommended_bets
        }
        for bet in display_bets:
            matched = selected_keys.get((bet.ticket_type, bet.combination), bet)
            final_bet = matched.recommended_bet if strategy.recommended_bets else 0.0
            recommendation_table.add_row(
                bet.ticket_type.value,
                format_combination(bet.ticket_type, bet.combination),
                f"{bet.probability:.3f}",
                f"{bet.odds_value:.2f}",
                f"{bet.expected_value:.3f}",
                f"{bet.stake_weight:.4f}",
                f"{final_bet:.0f}",
            )
    else:
        recommendation_table.add_row("-", "-", "-", "-", "-", "-", "-")
    console.print(recommendation_table)

    if strategy.skip_reason:
        console.print(f"[yellow]Skip reason: {strategy.skip_reason}[/yellow]")
    elif not market_odds:
        console.print("[yellow]No eligible market odds found before judgment time.[/yellow]")

    if save:
        with session_factory() as session:
            strategy_run = StrategyRun(
                race_id=bundle.race_id,
                model_version=bundle.model_version,
                evaluation_mode=EvaluationMode.LIVE.value,
                judgment_time=decision_time,
                stake_sizing_rule="min_100_normalized",
                confidence_score=strategy.confidence_score,
                skip_reason=strategy.skip_reason,
                total_recommended_bet=strategy.total_recommended_bet,
            )
            predictions = build_prediction_rows(bundle.race_id, bundle.model_version, decision_time, strategy)
            repo.save_strategy_run(session, strategy_run, predictions)
        console.print("[green]Strategy run saved.[/green]")


def _resolve_judgment_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(tzinfo=None)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def build_prediction_rows(
    race_id: int,
    model_version: str,
    decision_time: datetime,
    strategy: StrategyRunResult,
) -> list[Prediction]:
    selected_lookup = {
        (bet.ticket_type, bet.combination): bet
        for bet in strategy.recommended_bets
    }
    rows = []
    for bet in (strategy.candidate_bets or strategy.recommended_bets):
        selected = selected_lookup.get((bet.ticket_type, bet.combination))
        rows.append(
            Prediction(
                race_id=race_id,
                model_version=model_version,
                predicted_at=decision_time,
                ticket_type=bet.ticket_type.value,
                combination=format_combination(bet.ticket_type, bet.combination),
                predicted_prob=bet.probability,
                odds_at_prediction=bet.odds_value,
                expected_value=bet.expected_value,
                kelly_fraction=bet.kelly_fraction,
                stake_weight=bet.stake_weight,
                recommended_bet=selected.recommended_bet if selected is not None else 0.0,
                confidence_score=bet.confidence_score,
                skip_reason=bet.skip_reason if selected is not None else strategy.skip_reason,
            )
        )
    return rows


def get_missing_trial_positions(race_df: pl.DataFrame) -> list[int]:
    if "trial_time" not in race_df.columns:
        return []
    missing_df = race_df.filter(pl.col("trial_time").is_null()).sort("post_position")
    return [int(value) for value in missing_df["post_position"].to_list()]


async def _refresh_race_entries(
    target_date: date,
    track_code: TrackCode,
    race_no: int,
    repo: Repository,
    session_factory,
) -> str | None:
    settings = get_settings()
    try:
        async with AutoraceJpScraper(settings) as scraper:
            entries = await scraper.get_race_entries(track_code, target_date, race_no)
    except Exception as exc:  # noqa: BLE001
        return f"Warning: 最新 Program の取得に失敗したため、既存 DB データで予測します。({exc})"

    if not entries.entries:
        return "Warning: 最新 Program に出走情報が無いため、既存 DB データで予測します。"

    try:
        with session_factory() as session:
            repo.save_race_data(session, entries, None, update_race_metadata=False)
    except Exception as exc:  # noqa: BLE001
        return f"Warning: 最新 Program の保存に失敗したため、既存 DB データで予測します。({exc})"
    return None
