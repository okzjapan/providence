"""Retrain CLI command."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from providence.feedback.retrain import run_retrain_workflow

console = Console()


def retrain_command(
    compare_with: str = typer.Option("latest", "--compare-with", help="Baseline model version"),
    compare_window_days: int = typer.Option(28, "--compare-window-days", help="Recent backtest comparison window"),
    optimize: bool = typer.Option(False, "--optimize", help="Run hyperparameter search"),
    n_trials: int = typer.Option(30, "--n-trials", help="Number of Optuna trials"),
    shap_samples: int = typer.Option(1000, "--shap-samples", help="SHAP sample size"),
    rebuild_features: bool = typer.Option(False, "--rebuild-features", help="Rebuild feature cache"),
    promote: bool = typer.Option(False, "--promote", help="Promote candidate model to latest"),
) -> None:
    """Train candidate model and compare with baseline."""
    result = run_retrain_workflow(
        compare_with=compare_with,
        compare_window_days=compare_window_days,
        optimize=optimize,
        n_trials=n_trials,
        shap_samples=shap_samples,
        rebuild_features=rebuild_features,
        promote=promote,
    )

    summary = Table(title="Retrain Result")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Baseline", result.baseline_version)
    summary.add_row("Candidate", result.candidate_version)
    summary.add_row("Promoted", result.promoted_version or "-")
    summary.add_row("Candidate win_accuracy", _fmt(result.candidate_metrics.get("win_accuracy")))
    summary.add_row("Baseline win_accuracy", _fmt(result.baseline_metrics.get("win_accuracy")))
    summary.add_row("Candidate brier", _fmt(result.candidate_metrics.get("brier_score"), digits=6))
    summary.add_row("Baseline brier", _fmt(result.baseline_metrics.get("brier_score"), digits=6))
    summary.add_row("Candidate ROI (recent backtest)", _fmt(result.backtest_candidate_roi))
    summary.add_row("Baseline ROI (recent backtest)", _fmt(result.backtest_baseline_roi))
    console.print(summary)


def _fmt(value, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"
