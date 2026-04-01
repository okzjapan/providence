"""Prediction CLI command."""

from __future__ import annotations

from datetime import date

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore

console = Console()


def predict_command(
    date_str: str = typer.Option(..., "--date", help="Race date (YYYY-MM-DD)"),
    track: str = typer.Option(..., "--track", help="Track name"),
    race: int = typer.Option(..., "--race", help="Race number"),
) -> None:
    target_date = date.fromisoformat(date_str)
    track_code = TrackCode.from_name(track)
    loader = DataLoader()
    pipeline = FeaturePipeline()
    predictor = Predictor(ModelStore(), pipeline, loader)

    race_df = loader.load_race_dataset(start_date=target_date, end_date=target_date).filter(
        (pl.col("track_id") == track_code.value) & (pl.col("race_number") == race)
    )
    if race_df.is_empty():
        console.print("[red]対象レースのデータがありません。先に `providence scrape day` を実行してください。[/red]")
        raise typer.Exit(1)

    predictor.load_history(target_date)
    probs = predictor.predict_race(race_df)

    table = Table(title=f"{track_code.japanese_name} {target_date} R{race}")
    table.add_column("Car", justify="right")
    table.add_column("WinProb", justify="right")
    for car, prob in sorted(probs["win"].items()):
        table.add_row(str(car + 1), f"{prob:.3f}")
    console.print(table)
