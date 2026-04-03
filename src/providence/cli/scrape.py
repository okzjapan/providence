"""Data collection CLI commands."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime

import structlog
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.domain.enums import TrackCode
from providence.scraper.autorace_jp import AutoraceJpScraper
from providence.scraper.oddspark import OddsparkScraper

scrape_app = typer.Typer()
console = Console()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _resolve_track(name: str | None) -> TrackCode | None:
    if name is None:
        return None
    try:
        return TrackCode.from_name(name)
    except ValueError:
        console.print(f"[red]Unknown track: {name}[/red]")
        valid = ", ".join(t.name.lower() for t in TrackCode)
        console.print(f"[dim]Valid tracks: {valid}[/dim]")
        raise typer.Exit(1)


@scrape_app.command("today")
def scrape_today() -> None:
    """Show today's race schedule."""
    asyncio.run(_scrape_today())


async def _scrape_today() -> None:
    settings = get_settings()
    async with AutoraceJpScraper(settings) as scraper:
        data = await scraper.get_today_schedule()

    body = data.get("body", data)
    today_list = body.get("today", [])

    if not today_list:
        console.print("[yellow]No races scheduled today.[/yellow]")
        return

    table = Table(title="Today's Races")
    table.add_column("Track", style="cyan")
    table.add_column("Grade", style="green")
    table.add_column("On Sale Race", justify="right")
    table.add_column("Last Result", justify="right")
    table.add_column("Start")
    table.add_column("Vote Close")
    table.add_column("Weather")
    table.add_column("Track Cond.")
    table.add_column("Temp")

    for item in today_list:
        table.add_row(
            item.get("placeName", ""),
            item.get("gradeName", ""),
            str(item.get("nowRaceNo", "")),
            str(item.get("resultRaceNo", "")),
            item.get("raceStartTime", ""),
            item.get("telvoteTime", ""),
            item.get("weather", ""),
            "",
            f"{item.get('temp', '')}℃",
        )

    console.print(table)


@scrape_app.command("day")
def scrape_day(
    date_str: str = typer.Option(..., "--date", help="Date (YYYY-MM-DD)"),
    track_name: str | None = typer.Option(None, "--track", help="Track name (e.g. hamamatsu)"),
) -> None:
    """Scrape entries and results for a specific date."""
    asyncio.run(_scrape_day(date_str, track_name))


async def _scrape_day(date_str: str, track_name: str | None) -> None:
    race_date = _parse_date(date_str)
    track = _resolve_track(track_name)
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()

    with session_factory() as session:
        repo.ensure_tracks(session)

    tracks = [track] if track else list(TrackCode)
    total = 0
    errors = 0
    start = time.monotonic()
    log = structlog.get_logger()

    async with AutoraceJpScraper(settings) as scraper:
        for t in tracks:
            for race_no in range(1, 13):
                try:
                    entries = await scraper.get_race_entries(t, race_date, race_no)
                except Exception as exc:
                    log.debug("scrape_day_entry_skip", track=t.name, race_no=race_no, error=str(exc))
                    continue

                if not entries.entries:
                    continue

                result = None
                try:
                    result = await scraper.get_race_result(t, race_date, race_no)
                except Exception as exc:
                    log.warning("scrape_day_result_fail", track=t.name, race_no=race_no, error=str(exc))

                try:
                    with session_factory() as session:
                        repo.save_race_data(session, entries, result)
                        total += 1
                except Exception as exc:
                    errors += 1
                    log.error("scrape_day_save_fail", track=t.name, race_no=race_no, error=str(exc))

    # Fetch weather from oddspark for each track-date
    weather_count = 0
    async with OddsparkScraper(settings) as oddspark:
        for t in tracks:
            try:
                conditions = await oddspark.get_race_conditions(t, race_date)
                if any(v is not None for v in conditions.values()):
                    with session_factory() as session:
                        weather_count = repo.update_race_conditions(session, t.value, race_date, conditions)
            except Exception as exc:
                log.debug("weather_fetch_fail", track=t.name, error=str(exc))

    elapsed = time.monotonic() - start
    status = "success" if errors == 0 else "partial"
    console.print(f"[green]Saved {total} races, {weather_count} weather, {errors} errors in {elapsed:.1f}s[/green]")

    with session_factory() as session:
        repo.log_scrape(
            session,
            source="autorace_jp",
            target="results",
            target_date=race_date,
            records_count=total,
            status=status,
            duration_sec=elapsed,
        )


@scrape_app.command("players")
def scrape_players() -> None:
    """Collect all rider master data from oddspark."""
    asyncio.run(_scrape_players())


async def _scrape_players() -> None:
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()

    with session_factory() as session:
        repo.ensure_tracks(session)

    start = time.monotonic()

    async with OddsparkScraper(settings) as scraper:
        players = await scraper.get_all_players()

    with session_factory() as session:
        for p in players:
            repo.save_player(session, p)

    elapsed = time.monotonic() - start
    console.print(f"[green]Saved {len(players)} players in {elapsed:.1f}s[/green]")

    with session_factory() as session:
        repo.log_scrape(
            session,
            source="oddspark",
            target="players",
            records_count=len(players),
            status="success",
            duration_sec=elapsed,
        )


@scrape_app.command("odds")
def scrape_odds(
    date_str: str = typer.Option(..., "--date", help="Date (YYYY-MM-DD)"),
    track_name: str = typer.Option(..., "--track", help="Track name"),
    race_no: int | None = typer.Option(None, "--race", help="Race number"),
) -> None:
    """Collect market odds snapshots for one track/day."""
    asyncio.run(_scrape_odds(date_str, track_name, race_no))


async def _scrape_odds(date_str: str, track_name: str, race_no: int | None) -> None:
    race_date = _parse_date(date_str)
    track = _resolve_track(track_name)
    if track is None:
        raise typer.Exit(1)

    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()
    target_races = [race_no] if race_no is not None else list(range(1, 13))
    saved = 0
    errors = 0
    start = time.monotonic()

    async with AutoraceJpScraper(settings) as scraper:
        for target_race in target_races:
            try:
                entries = await scraper.get_race_entries(track, race_date, target_race)
                odds = await scraper.get_odds(track, race_date, target_race)
            except Exception as exc:
                errors += 1
                structlog.get_logger().warning(
                    "scrape_odds_fail",
                    track=track.name,
                    race_date=str(race_date),
                    race_no=target_race,
                    error=str(exc),
                )
                continue

            if not odds.odds:
                continue

            with session_factory() as session:
                race = repo.save_race_data(session, entries, None) if entries.entries else repo.get_race(
                    session, track.value, race_date, target_race
                )
                if race is None:
                    continue
                saved += repo.save_odds(
                    session,
                    race.id,
                    odds.odds,
                    source_name="autorace_jp",
                )

    elapsed = time.monotonic() - start
    console.print(f"[green]Saved {saved} odds rows with {errors} errors in {elapsed:.1f}s[/green]")
    with session_factory() as session:
        repo.log_scrape(
            session,
            source="autorace_jp",
            target="odds",
            target_date=race_date,
            track_id=track.value,
            records_count=saved,
            status="success" if errors == 0 else "partial",
            duration_sec=elapsed,
        )


@scrape_app.command("results")
def scrape_results(
    date_str: str = typer.Option(..., "--date", help="Date (YYYY-MM-DD)"),
    track_name: str | None = typer.Option(None, "--track", help="Track name (e.g. hamamatsu)"),
) -> None:
    """Refresh race results for a specific date after races complete."""
    asyncio.run(_scrape_results(date_str, track_name))


async def _scrape_results(date_str: str, track_name: str | None) -> None:
    race_date = _parse_date(date_str)
    track = _resolve_track(track_name)
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()
    log = structlog.get_logger()

    tracks = [track] if track else list(TrackCode)
    total = 0
    errors = 0
    start = time.monotonic()

    async with AutoraceJpScraper(settings) as scraper:
        for t in tracks:
            for race_no in range(1, 13):
                try:
                    result = await scraper.get_race_result(t, race_date, race_no)
                    if not result.results:
                        continue
                    entries = await scraper.get_race_entries(t, race_date, race_no)
                except Exception as exc:
                    errors += 1
                    log.warning("scrape_results_fetch_fail", track=t.name, race_no=race_no, error=str(exc))
                    continue

                try:
                    with session_factory() as session:
                        repo.save_race_data(session, entries, result)
                        total += 1
                except Exception as exc:
                    errors += 1
                    log.error("scrape_results_save_fail", track=t.name, race_no=race_no, error=str(exc))

    elapsed = time.monotonic() - start
    status = "success" if errors == 0 else "partial"
    console.print(f"[green]Refreshed {total} race results with {errors} errors in {elapsed:.1f}s[/green]")
    with session_factory() as session:
        repo.log_scrape(
            session,
            source="autorace_jp",
            target="results_refresh",
            target_date=race_date,
            track_id=track.value if track else None,
            records_count=total,
            status=status,
            duration_sec=elapsed,
        )


@scrape_app.command("historical")
def scrape_historical(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    track_name: str | None = typer.Option(None, "--track", help="Track name"),
    resume: bool = typer.Option(False, "--resume", help="Skip already collected races"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show targets without collecting"),
) -> None:
    """Collect historical race data."""
    asyncio.run(_scrape_historical(from_date, to_date, track_name, resume, dry_run))


async def _scrape_historical(
    from_str: str,
    to_str: str,
    track_name: str | None,
    resume: bool,
    dry_run: bool,
) -> None:
    start_date = _parse_date(from_str)
    end_date = _parse_date(to_str)
    track = _resolve_track(track_name)
    settings = get_settings()
    session_factory = get_session_factory(settings)
    repo = Repository()
    log = structlog.get_logger()

    with session_factory() as session:
        repo.ensure_tracks(session)

    tracks = [track] if track else list(TrackCode)
    collected: set[tuple[int, date, int]] = set()

    if resume:
        with session_factory() as session:
            collected = repo.get_collected_races(
                session, start_date, end_date, track.value if track else None
            )
        console.print(f"[dim]Resume mode: {len(collected)} races already collected[/dim]")

    # Step 1: Discover racing dates via SearchRace
    console.print("[dim]Discovering racing dates...[/dim]")
    racing_dates: dict[TrackCode, list[date]] = {}

    async with AutoraceJpScraper(settings) as scraper:
        for t in tracks:
            try:
                results = await scraper.search_races([t], start_date, end_date)
                dates = set()
                for hold in results:
                    for d in hold.get("raceDateList", []):
                        try:
                            dates.add(_parse_date(d) if isinstance(d, str) else d)
                        except (ValueError, TypeError):
                            pass
                racing_dates[t] = sorted(dates)
            except Exception as e:
                console.print(f"[yellow]Could not discover dates for {t.japanese_name}: {e}[/yellow]")
                racing_dates[t] = []

    total_target = sum(len(dates) * 12 for dates in racing_dates.values())
    skip_count = len(collected)

    if dry_run:
        racing_day_count = sum(len(d) for d in racing_dates.values())
        console.print(f"\n[bold]Targets: ~{total_target} races across {racing_day_count} racing days[/bold]")
        console.print(f"[dim]Would skip {skip_count} already collected[/dim]")
        for t, dates in racing_dates.items():
            if dates:
                console.print(f"  {t.japanese_name}: {len(dates)} days ({dates[0]} ~ {dates[-1]})")
            else:
                console.print(f"  {t.japanese_name}: 0 days")
        return

    # Step 2: Collect data
    total_saved = 0
    total_errors = 0
    start_time = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting...", total=total_target)

        async with AutoraceJpScraper(settings) as scraper:
            for t in tracks:
                for race_date in racing_dates.get(t, []):
                    for race_no in range(1, 13):
                        progress.update(task, advance=1, description=f"{t.japanese_name} {race_date} R{race_no}")

                        if (t.value, race_date, race_no) in collected:
                            continue

                        try:
                            result = await scraper.get_race_result(t, race_date, race_no)
                            if not result.results:
                                continue

                            entries = await scraper.get_race_entries(t, race_date, race_no)

                            with session_factory() as session:
                                repo.save_race_data(session, entries, result)
                                total_saved += 1
                        except Exception as exc:
                            total_errors += 1
                            log.warning(
                                "historical_race_fail",
                                track=t.name, date=str(race_date), race_no=race_no,
                                error=str(exc),
                            )

    # Step 3: Fetch weather/conditions from oddspark (1 request per track-date)
    weather_count = 0
    console.print("[dim]Fetching weather/track conditions from oddspark...[/dim]")
    async with OddsparkScraper(settings) as oddspark:
        for t in tracks:
            for race_date in racing_dates.get(t, []):
                try:
                    conditions = await oddspark.get_race_conditions(t, race_date)
                    if any(v is not None for v in conditions.values()):
                        with session_factory() as session:
                            updated = repo.update_race_conditions(session, t.value, race_date, conditions)
                            weather_count += updated
                except Exception as exc:
                    log.debug("weather_fetch_fail", track=t.name, date=str(race_date), error=str(exc))

    elapsed = time.monotonic() - start_time
    console.print(
        f"\n[green]Saved {total_saved} races, {weather_count} weather updates, "
        f"{total_errors} errors, {elapsed:.0f}s elapsed[/green]"
    )

    with session_factory() as session:
        repo.log_scrape(
            session,
            source="autorace_jp",
            target="results",
            target_date=start_date,
            records_count=total_saved,
            status="success" if total_errors == 0 else "partial",
            duration_sec=elapsed,
        )


@scrape_app.command("status")
def scrape_status() -> None:
    """Show recent scrape log entries."""
    session_factory = get_session_factory()
    from sqlalchemy import select

    from providence.database.tables import ScrapeLog

    with session_factory() as session:
        logs = session.execute(
            select(ScrapeLog).order_by(ScrapeLog.executed_at.desc()).limit(20)
        ).scalars().all()

    if not logs:
        console.print("[yellow]No scrape logs found.[/yellow]")
        return

    table = Table(title="Recent Scrape Logs")
    table.add_column("Time", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Target")
    table.add_column("Date")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Status")
    table.add_column("Duration", justify="right")

    for log in logs:
        status_style = "green" if log.status == "success" else "yellow" if log.status == "partial" else "red"
        table.add_row(
            str(log.executed_at)[:19] if log.executed_at else "",
            log.source,
            log.target,
            str(log.target_date) if log.target_date else "",
            str(log.records_count),
            f"[{status_style}]{log.status}[/{status_style}]",
            f"{log.duration_sec:.1f}s" if log.duration_sec else "",
        )

    console.print(table)
