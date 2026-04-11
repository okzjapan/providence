"""Autobet CLI: automated prediction + Slack notification commands."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import typer
from rich.console import Console
from rich.table import Table

from providence.cli.strategy_options import build_strategy_config
from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.services.autobet import run_autobet_tick, run_daily_overview, run_morning_sync

autobet_app = typer.Typer()
console = Console()

JST = timezone(timedelta(hours=9))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@autobet_app.command("morning-sync")
def morning_sync_command(
    date_str: str = typer.Option("", "--date", help="Date (YYYY-MM-DD). Defaults to today (JST)."),
    track_name: str | None = typer.Option(None, "--track", help="Track name (e.g. 川口). Defaults to all tracks."),
    skip_overview: bool = typer.Option(False, "--skip-overview", help="Skip daily overview notification"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip Slack notification"),
    model_version: str = typer.Option("latest", "--model-version"),
) -> None:
    """Scrape today's race data, sync start times, and send daily overview."""
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()
    loader = DataLoader()

    target_date = _parse_date(date_str) if date_str else None
    tracks = [TrackCode.from_name(track_name)] if track_name else None

    run_morning_sync(
        session_factory=session_factory,
        repo=repo,
        loader=loader,
        target_date=target_date,
        tracks=tracks,
        settings=settings,
    )
    console.print("[green]Morning sync complete.[/green]")

    if not skip_overview:
        config = build_strategy_config(
            ticket_types=PROVEN_STRATEGY_DEFAULTS["ticket_types"],
            max_candidates=PROVEN_STRATEGY_DEFAULTS["max_candidates"],
            fractional_kelly=PROVEN_STRATEGY_DEFAULTS["fractional_kelly"],
            min_confidence=PROVEN_STRATEGY_DEFAULTS["min_confidence"],
            min_expected_value=PROVEN_STRATEGY_DEFAULTS["min_expected_value"],
        )
        predictor = Predictor(ModelStore(), FeaturePipeline(), loader, version=model_version)
        webhook_url = settings.slack_webhook_url if not dry_run else None
        run_daily_overview(
            session_factory=session_factory,
            repo=repo,
            predictor=predictor,
            loader=loader,
            config=config,
            settings=settings,
            slack_webhook_url=webhook_url,
            target_date=_parse_date(date_str) if date_str else None,
            dry_run=dry_run,
        )
        console.print("[green]Daily overview complete.[/green]")


PROVEN_STRATEGY_DEFAULTS = {
    "ticket_types": "win,wide",
    "fractional_kelly": 0.05,
    "min_confidence": 0.90,
    "min_expected_value": 0.40,
    "max_candidates": 2,
}


@autobet_app.command("tick")
def tick_command(
    lead_minutes: float = typer.Option(10.0, "--lead-minutes", help="Minutes before deadline to trigger"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip Slack notification, log to console only"),
    force_race_track: str | None = typer.Option(None, "--force-race", help="Force a specific track (e.g. 川口)"),
    force_race_number: int | None = typer.Option(None, "--force-race-number", help="Force a specific race number"),
    model_version: str = typer.Option("latest", "--model-version", help="Model version to use"),
    save: bool = typer.Option(True, "--save/--no-save", help="Persist prediction to DB"),
    fractional_kelly: float | None = typer.Option(None, "--fractional-kelly", help="Default: 0.05 (proven)"),
    min_confidence: float | None = typer.Option(None, "--min-confidence", help="Default: 0.90 (proven)"),
    min_expected_value: float | None = typer.Option(None, "--min-expected-value", help="Default: 0.40 (proven)"),
    ticket_types: str | None = typer.Option(None, "--ticket-types", help="Default: win (proven)"),
    max_candidates: int | None = typer.Option(None, "--max-candidates", help="Default: 2 (proven)"),
    slack_mention: str = typer.Option(
        "<@U070Q54JAAC>", "--slack-mention", help="Mention for S-rank bets"
    ),
) -> None:
    """Run prediction for approaching races and send Slack notifications.

    Default parameters are the proven strategy (V013 backtest ROI +54.5%):
    win only, kelly=0.05, conf>=0.90, EV>=0.40, cand<=2.
    """
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()
    loader = DataLoader()
    predictor = Predictor(ModelStore(), FeaturePipeline(), loader, version=model_version)
    config = build_strategy_config(
        ticket_types=ticket_types or PROVEN_STRATEGY_DEFAULTS["ticket_types"],
        max_candidates=max_candidates or PROVEN_STRATEGY_DEFAULTS["max_candidates"],
        fractional_kelly=fractional_kelly or PROVEN_STRATEGY_DEFAULTS["fractional_kelly"],
        min_confidence=min_confidence or PROVEN_STRATEGY_DEFAULTS["min_confidence"],
        min_expected_value=min_expected_value or PROVEN_STRATEGY_DEFAULTS["min_expected_value"],
    )

    force_race = None
    if force_race_track and force_race_number:
        force_race = (TrackCode.from_name(force_race_track), force_race_number)
    elif force_race_track or force_race_number:
        console.print("[red]Both --force-race and --force-race-number must be specified together.[/red]")
        raise typer.Exit(1)

    webhook_url = settings.slack_webhook_url if not dry_run else None

    results = run_autobet_tick(
        session_factory=session_factory,
        repo=repo,
        predictor=predictor,
        loader=loader,
        config=config,
        settings=settings,
        slack_webhook_url=webhook_url,
        slack_mention=slack_mention,
        lead_minutes=lead_minutes,
        save=save,
        dry_run=dry_run,
        force_race=force_race,
    )

    if not results:
        console.print("[dim]No races within deadline window.[/dim]")
        return

    table = Table(title="Tick Results")
    table.add_column("Track")
    table.add_column("Race", justify="right")
    table.add_column("Status")
    table.add_column("Bets?")
    table.add_column("Total", justify="right")
    table.add_column("Slack")

    for r in results:
        table.add_row(
            r.track_name,
            str(r.race_number),
            r.status,
            "✓" if r.has_recommended_bets else "-",
            f"¥{r.total_recommended_bet:,.0f}" if r.has_recommended_bets else "-",
            "✓" if r.slack_sent else "-",
        )
    console.print(table)


@autobet_app.command("calibrate-model-b")
def calibrate_model_b_command(
    model_version: str = typer.Option("latest", "--model-version"),
    start_date: str = typer.Option(..., "--start", help="Calibration start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", help="Calibration end date (YYYY-MM-DD)"),
    percentile: float = typer.Option(0.30, "--percentile", help="Percentile threshold (0.30 = keep top 70%)"),
    alpha: float = typer.Option(0.5, "--alpha", help="Model A/B blending weight"),
) -> None:
    """Calibrate Model B race quality threshold from historical data."""
    from providence.model.model_b_filter import ModelBFilter
    from providence.model.store import ModelStore

    settings = get_settings()
    get_session_factory(settings)
    loader = DataLoader()
    store = ModelStore()

    version_dir = store.version_dir(model_version)
    model_b_filter = ModelBFilter.load(version_dir, alpha=alpha)
    if model_b_filter is None:
        console.print("[red]Model B not found in this version.[/red]")
        raise typer.Exit(1)

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    console.print(f"Loading features from {start} to {end}...")
    features = loader.load_race_dataset(start_date=start, end_date=end)
    console.print(f"Loaded {features.height} rows, {features['race_id'].n_unique()} races")

    if model_b_filter.model_a_version:
        model_a_booster, model_a_meta = store.load(model_b_filter.model_a_version)
    else:
        model_a_booster, model_a_meta = store.load(model_version)
    model_a_features = model_a_meta["feature_columns"]

    console.print("Computing race quality scores...")
    threshold = ModelBFilter.calibrate_threshold(
        features, model_a_booster, model_a_features, model_b_filter, percentile=percentile,
    )

    model_b_filter.save_threshold(threshold, version_dir)
    console.print(f"[green]Threshold saved: {threshold:.6f} (top {(1-percentile)*100:.0f}% pass)[/green]")


@autobet_app.command("status")
def status_command(
    date_str: str = typer.Option("", "--date", help="Date (YYYY-MM-DD). Defaults to today (JST)."),
) -> None:
    """Show today's processed races and notification status."""
    settings = get_settings()
    session_factory = get_session_factory(settings)

    target_date = _parse_date(date_str) if date_str else datetime.now(tz=JST).date()

    from sqlalchemy import select

    from providence.database.tables import Race, StrategyRun

    with session_factory() as session:
        runs = list(
            session.execute(
                select(StrategyRun, Race)
                .join(Race, StrategyRun.race_id == Race.id)
                .where(
                    Race.race_date == target_date,
                    StrategyRun.evaluation_mode == "live",
                )
                .order_by(Race.track_id, Race.race_number)
            ).all()
        )

    if not runs:
        console.print(f"[dim]No autobet runs for {target_date}.[/dim]")
        return

    table = Table(title=f"Autobet Status: {target_date}")
    table.add_column("Track")
    table.add_column("Race", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Skip Reason")
    table.add_column("Total Bet", justify="right")
    table.add_column("Time")

    for run, race in runs:
        track_name = TrackCode(race.track_id).japanese_name
        table.add_row(
            track_name,
            str(race.race_number),
            f"{run.confidence_score:.2f}" if run.confidence_score else "-",
            run.skip_reason or "-",
            f"¥{run.total_recommended_bet:,.0f}" if run.total_recommended_bet else "-",
            run.judgment_time.strftime("%H:%M") if run.judgment_time else "-",
        )
    console.print(table)
