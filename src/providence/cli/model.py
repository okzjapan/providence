"""Model management CLI commands."""

from __future__ import annotations

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from providence.model.store import ModelStore

model_app = typer.Typer()
console = Console()


@model_app.command("list")
def list_models() -> None:
    store = ModelStore()
    versions = store.list_versions()
    table = Table(title="Model Versions")
    table.add_column("Version")
    table.add_column("Created")
    table.add_column("Type")
    table.add_column("WinAcc")
    table.add_column("Gate")
    for item in versions:
        metrics = item.get("metrics", {})
        gate = item.get("gate", {})
        table.add_row(
            item.get("version", ""),
            item.get("created_at", ""),
            item.get("model_type", ""),
            f"{metrics.get('win_accuracy', float('nan')):.3f}" if "win_accuracy" in metrics else "",
            "PASS" if gate.get("passed") else "FAIL" if gate else "",
        )
    console.print(table)


@model_app.command("evaluate")
def show_model(version: str = typer.Option("latest", "--version")) -> None:
    store = ModelStore()
    _, metadata = store.load(version)
    metrics = metadata.get("metrics", {})
    gate = metadata.get("gate", {})

    summary = Table(title=f"Model Evaluation: {metadata.get('version', version)}")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("model_type", str(metadata.get("model_type", "")))
    summary.add_row("temperature", f"{metadata.get('temperature', float('nan')):.4f}")
    summary.add_row("win_accuracy", f"{metrics.get('win_accuracy', float('nan')):.4f}")
    summary.add_row("top3_overlap", f"{metrics.get('top3_overlap', float('nan')):.4f}")
    summary.add_row("brier_score", f"{metrics.get('brier_score', float('nan')):.6f}")
    summary.add_row("brier_baseline", f"{metrics.get('brier_baseline', float('nan')):.6f}")
    if not gate:
        gate_label = "N/A"
    else:
        gate_label = "PASS" if gate.get("passed") else "FAIL"
    summary.add_row("gate", gate_label)
    console.print(summary)

    if metadata.get("split"):
        split_table = Table(title="Split")
        split_table.add_column("Name")
        split_table.add_column("Start")
        split_table.add_column("End")
        for name, value in metadata["split"].items():
            split_table.add_row(name, value[0], value[1])
        console.print(split_table)

    version_dir = store.version_dir(version)
    importance_path = version_dir / "feature_importance.csv"
    shap_path = version_dir / "shap_importance.csv"
    if importance_path.exists():
        df = pl.read_csv(importance_path).head(10)
        table = Table(title="Top Feature Importance (gain)")
        table.add_column("Feature")
        table.add_column("Gain", justify="right")
        for row in df.iter_rows():
            table.add_row(str(row[0]), f"{float(row[1]):.2f}")
        console.print(table)
    if shap_path.exists():
        df = pl.read_csv(shap_path).head(10)
        table = Table(title="Top SHAP Importance")
        table.add_column("Feature")
        table.add_column("MeanAbsSHAP", justify="right")
        for row in df.iter_rows():
            table.add_row(str(row[0]), f"{float(row[1]):.6f}")
        console.print(table)
