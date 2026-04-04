"""Replay listing and detail CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from providence.database.engine import get_session_factory
from providence.database.repository import Repository

console = Console()

replay_app = typer.Typer(help="Historical replay management")


@replay_app.command("list")
def list_replays(
    limit: int = typer.Option(20, "--limit", help="Max rows to show"),
) -> None:
    """List saved simulation runs."""
    session_factory = get_session_factory()
    repo = Repository()
    with session_factory() as session:
        runs = repo.get_simulation_runs(session, limit=limit)

    table = Table(title="Simulation Runs")
    table.add_column("ID", justify="right")
    table.add_column("Model")
    table.add_column("Mode")
    table.add_column("Odds")
    table.add_column("Stake Rule")
    table.add_column("Period")
    table.add_column("Races", justify="right")
    table.add_column("Eval", justify="right")
    table.add_column("ROI", justify="right")
    table.add_column("Created")

    if not runs:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
    else:
        for run in runs:
            table.add_row(
                str(run.id),
                run.model_version,
                run.evaluation_mode,
                run.odds_policy,
                run.stake_sizing_rule,
                f"{run.date_range_start} ~ {run.date_range_end}",
                str(run.total_races),
                str(run.evaluated_races),
                f"{run.roi:.3f}",
                str(run.created_at)[:19],
            )
    console.print(table)


@replay_app.command("detail")
def replay_detail(
    run_id: int = typer.Argument(..., help="Simulation run ID"),
) -> None:
    """Show race-level detail for a simulation run."""
    session_factory = get_session_factory()
    repo = Repository()
    with session_factory() as session:
        strategy_runs = repo.get_strategy_runs_for_simulation(session, run_id)
        run_ids = [int(sr.id) for sr in strategy_runs]
        predictions = repo.get_predictions_for_strategy_runs(session, run_ids)
        betting_logs = repo.get_betting_logs_for_prediction_ids(session, [int(p.id) for p in predictions])

    logs_by_pred = {int(bl.prediction_id): bl for bl in betting_logs}

    summary_table_obj = Table(title=f"Simulation Run #{run_id} — Race Detail")
    summary_table_obj.add_column("Race", justify="right")
    summary_table_obj.add_column("Skip")
    summary_table_obj.add_column("Bets", justify="right")
    summary_table_obj.add_column("Stake", justify="right")
    summary_table_obj.add_column("Payout", justify="right")
    summary_table_obj.add_column("Profit", justify="right")

    for sr in strategy_runs:
        sr_preds = [p for p in predictions if p.strategy_run_id == sr.id]
        sr_logs = [logs_by_pred[int(p.id)] for p in sr_preds if int(p.id) in logs_by_pred]
        bet_count = sum(1 for p in sr_preds if (p.recommended_bet or 0) > 0)
        stake = sum(bl.actual_bet_amount for bl in sr_logs)
        payout = sum(bl.payout for bl in sr_logs)
        profit = sum(bl.profit for bl in sr_logs)
        summary_table_obj.add_row(
            str(sr.race_id),
            sr.skip_reason or "-",
            str(bet_count),
            f"{stake:.0f}",
            f"{payout:.0f}",
            f"{profit:.0f}",
        )

    console.print(summary_table_obj)
