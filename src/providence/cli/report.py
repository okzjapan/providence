"""Operational report CLI command."""

from __future__ import annotations

from datetime import date, datetime

import typer
from rich.console import Console
from rich.table import Table

from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.feedback.drift import detect_drift
from providence.feedback.performance import refresh_model_performance
from providence.feedback.reconcile import reconcile_paper_trades
from providence.model.store import ModelStore

console = Console()


def report_command(
    refresh: bool = typer.Option(False, "--refresh", help="Refresh reconcile/performance before reporting"),
) -> None:
    """Show operational report with optional refresh."""
    session_factory = get_session_factory()
    repo = Repository()
    store = ModelStore()
    drift_result = None
    refresh_result: dict[str, object] = {}

    if refresh:
        with session_factory() as session:
            reconcile_result = reconcile_paper_trades(session, repository=repo)
            repo.log_feedback_run(
                session,
                job_name="reconcile",
                evaluation_date=reconcile_result.reconciled_at.date(),
                status="success",
                details=f"runs={reconcile_result.strategy_runs},logs={reconcile_result.betting_logs_written}",
                executed_at=reconcile_result.reconciled_at,
            )
            strategy_summary = repo.get_strategy_run_summary(session)
            latest_run = strategy_summary.get("latest_run")
            evaluation_date = latest_run.judgment_time.date() if latest_run is not None else date.today()
            performance_result = refresh_model_performance(session, end_date=evaluation_date, repository=repo)
            latest_model_version = _resolve_latest_model_version(store, latest_run)
            repo.log_feedback_run(
                session,
                job_name="performance",
                model_version=latest_model_version,
                evaluation_date=evaluation_date,
                status="success",
                details=f"rows={performance_result.rows_written}",
                executed_at=performance_result.computed_at,
            )
            if latest_model_version is not None:
                drift_result = detect_drift(
                    session,
                    model_version=latest_model_version,
                    evaluation_date=evaluation_date,
                    repository=repo,
                    store=store,
                )
                repo.log_feedback_run(
                    session,
                    job_name="drift",
                    model_version=latest_model_version,
                    evaluation_date=evaluation_date,
                    status="warning" if drift_result.warnings else "success",
                    details=",".join(drift_result.warnings) if drift_result.warnings else None,
                    executed_at=datetime.now(),
                )
            refresh_result = {
                "reconcile_runs": reconcile_result.strategy_runs,
                "reconcile_logs": reconcile_result.betting_logs_written,
                "performance_rows": performance_result.rows_written,
                "drift_warnings": len(drift_result.warnings) if drift_result is not None else 0,
            }

    with session_factory() as session:
        scrape_logs = repo.get_recent_scrape_logs(session, limit=10)
        strategy_summary = repo.get_strategy_run_summary(session)
        performance_rows = repo.get_recent_model_performance(session, limit=12)
        freshness = repo.get_feedback_freshness(session)

    if refresh:
        console.print(_refresh_table(refresh_result))
    console.print(_strategy_table(strategy_summary))
    console.print(_model_table(store, strategy_summary))
    console.print(_freshness_table(freshness, drift_result))
    console.print(_performance_table(performance_rows))
    if drift_result is not None:
        console.print(_drift_table(drift_result))
    console.print(_scrape_log_table(scrape_logs))


def _strategy_table(summary: dict[str, object]) -> Table:
    latest_run = summary.get("latest_run")
    table = Table(title="Strategy Operations")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Strategy runs", str(summary.get("strategy_runs", 0)))
    table.add_row("Prediction legs", str(summary.get("predictions", 0)))
    table.add_row("Latest prediction", _fmt_dt(summary.get("latest_prediction_at")))
    if latest_run is not None:
        table.add_row("Latest model", str(getattr(latest_run, "model_version", "")))
        table.add_row("Latest mode", str(getattr(latest_run, "evaluation_mode", "")))
        table.add_row("Latest skip", str(getattr(latest_run, "skip_reason", "") or "-"))
        table.add_row("Latest total bet", f"{float(getattr(latest_run, 'total_recommended_bet', 0.0)):.0f}")
    else:
        table.add_row("Latest run", "none")
    return table


def _refresh_table(summary: dict[str, object]) -> Table:
    table = Table(title="Refresh Result")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Reconcile runs", str(summary.get("reconcile_runs", 0)))
    table.add_row("Betting logs written", str(summary.get("reconcile_logs", 0)))
    table.add_row("Performance rows", str(summary.get("performance_rows", 0)))
    table.add_row("Drift warnings", str(summary.get("drift_warnings", 0)))
    return table


def _model_table(store: ModelStore, strategy_summary: dict[str, object]) -> Table:
    try:
        latest = store.latest_metadata()
    except FileNotFoundError:
        latest = None
    table = Table(title="Latest Model")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    if latest is None:
        table.add_row("status", "none")
        return table

    latest_run = strategy_summary.get("latest_run")
    metrics = latest.get("metrics", {})
    gate = latest.get("gate", {})
    table.add_row("Version", str(latest.get("version", "")))
    table.add_row("Created", str(latest.get("created_at", "")))
    table.add_row("Type", str(latest.get("model_type", "")))
    table.add_row("Win accuracy", _fmt_float(metrics.get("win_accuracy")))
    table.add_row("Top3 overlap", _fmt_float(metrics.get("top3_overlap")))
    table.add_row("Brier", _fmt_float(metrics.get("brier_score"), digits=6))
    table.add_row("Gate", "PASS" if gate.get("passed") else "FAIL" if gate else "N/A")
    if latest_run is not None:
        table.add_row("Latest run model", str(getattr(latest_run, "model_version", "")))
    return table


def _freshness_table(freshness: dict[str, object], drift_result=None) -> Table:
    table = Table(title="Feedback Freshness")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Latest reconcile", _fmt_dt(freshness.get("latest_reconcile_at")))
    table.add_row("Latest performance", _fmt_dt(freshness.get("latest_performance_at")))
    if freshness.get("latest_drift_at") is not None:
        table.add_row("Latest drift check", _fmt_dt(freshness.get("latest_drift_at")))
    else:
        table.add_row("Latest drift check", str(getattr(drift_result, "checked_at", "-")))
    return table


def _performance_table(rows: list) -> Table:
    table = Table(title="Recent Model Performance")
    table.add_column("Date")
    table.add_column("Model")
    table.add_column("Window")
    table.add_column("Sample(win)", justify="right")
    table.add_column("WinAcc", justify="right")
    table.add_column("Top3", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("ROI", justify="right")
    if not rows:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-")
        return table
    for row in rows[:9]:
        table.add_row(
            str(getattr(row, "evaluation_date", "")),
            str(getattr(row, "model_version", "")),
            str(getattr(row, "window", "")),
            str(getattr(row, "sample_size", 0)),
            _fmt_float(getattr(row, "win_accuracy", None)),
            _fmt_float(getattr(row, "top3_accuracy", None)),
            _fmt_float(getattr(row, "brier_score", None), digits=6),
            _fmt_float(getattr(row, "roi", None)),
        )
    return table


def _drift_table(result) -> Table:
    table = Table(title="Drift Check")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Checked at", str(result.checked_at))
    table.add_row("Warnings", ", ".join(result.warnings) if result.warnings else "none")
    for key, value in sorted(result.metrics.items()):
        table.add_row(key, _fmt_float(value))
    if result.psi_scores:
        worst_feature = max(result.psi_scores, key=result.psi_scores.get)
        table.add_row("Worst PSI feature", f"{worst_feature} ({result.psi_scores[worst_feature]:.4f})")
    return table


def _scrape_log_table(logs: list) -> Table:
    table = Table(title="Recent Scrape Logs")
    table.add_column("Time", style="dim")
    table.add_column("Source")
    table.add_column("Target")
    table.add_column("Date")
    table.add_column("Rows", justify="right")
    table.add_column("Status")
    if not logs:
        table.add_row("-", "-", "-", "-", "-", "none")
        return table

    for log in logs:
        table.add_row(
            _fmt_dt(getattr(log, "executed_at", None), short=True),
            str(getattr(log, "source", "")),
            str(getattr(log, "target", "")),
            str(getattr(log, "target_date", "") or "-"),
            str(getattr(log, "records_count", 0)),
            str(getattr(log, "status", "")),
        )
    return table


def _fmt_dt(value, *, short: bool = False) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text[:19] if short else text


def _fmt_float(value, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _resolve_latest_model_version(store: ModelStore, latest_run) -> str | None:
    run_version = str(getattr(latest_run, "model_version", "") or "")
    if run_version:
        return run_version
    try:
        return store.latest_version()
    except FileNotFoundError:
        return None
