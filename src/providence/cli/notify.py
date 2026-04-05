"""Notification batch CLI commands."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone

import typer
from rich.console import Console
from rich.table import Table

from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TrackCode
from providence.scraper.oddspark import OddsparkScraper
from providence.services.schedule_resolver import find_approaching_deadlines
from providence.services.schedule_sync import sync_race_schedule

notify_app = typer.Typer()
console = Console()

JST = timezone(timedelta(hours=9))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@notify_app.command("schedule-sync")
def schedule_sync_command(
    date_str: str = typer.Option(
        "",
        "--date",
        help="Date (YYYY-MM-DD). Defaults to today (JST).",
    ),
    track_name: str | None = typer.Option(None, "--track", help="Track name"),
) -> None:
    """Sync race start times from oddspark into DB."""
    asyncio.run(_schedule_sync(date_str, track_name))


async def _schedule_sync(date_str: str, track_name: str | None) -> None:
    target_date = _parse_date(date_str) if date_str else datetime.now(tz=JST).date()
    tracks: list[TrackCode] | None = None
    if track_name:
        tracks = [TrackCode.from_name(track_name)]

    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()

    with session_factory() as session:
        repo.ensure_tracks(session)

    start = time.monotonic()
    async with OddsparkScraper(settings) as scraper:
        with session_factory() as session:
            result = await sync_race_schedule(
                session, scraper, target_date, tracks=tracks
            )
    elapsed = time.monotonic() - start

    console.print(
        f"[green]Schedule sync: {result.updated} updated, "
        f"{result.skipped} skipped, {result.errors} errors "
        f"in {elapsed:.1f}s[/green]"
    )

    with session_factory() as session:
        repo.log_scrape(
            session,
            source="oddspark",
            target="schedule_sync",
            target_date=target_date,
            records_count=result.updated,
            status="success" if result.errors == 0 else "partial",
            duration_sec=elapsed,
        )


@notify_app.command("deadline-scan")
def deadline_scan_command(
    lead_minutes: float = typer.Option(5.0, "--lead-minutes", help="Minutes before deadline"),
) -> None:
    """Show races approaching their voting deadline."""
    settings = get_settings()
    session_factory = get_session_factory(settings)
    now = datetime.now(tz=JST).replace(tzinfo=None)

    with session_factory() as session:
        candidates = find_approaching_deadlines(session, now=now, lead_minutes=lead_minutes)

    if not candidates:
        console.print("[yellow]No races within deadline window.[/yellow]")
        return

    table = Table(title=f"Races within {lead_minutes} min of deadline")
    table.add_column("Track", style="cyan")
    table.add_column("Race", justify="right")
    table.add_column("Deadline", style="red")
    table.add_column("Start", style="green")
    table.add_column("Min Left", justify="right")

    for c in candidates:
        track_name = TrackCode(c.track_id).japanese_name
        table.add_row(
            track_name,
            str(c.race_number),
            c.telvote_close_at.strftime("%H:%M"),
            c.scheduled_start_at.strftime("%H:%M") if c.scheduled_start_at else "-",
            f"{c.minutes_until_close:.1f}",
        )

    console.print(table)
