"""Schedule sync service: fetch all race start times and persist to DB."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import NamedTuple

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from providence.database.tables import Race
from providence.domain.enums import TrackCode
from providence.scraper.oddspark import OddsparkScraper

JST = timezone(timedelta(hours=9))

TELVOTE_OFFSET = timedelta(minutes=2)


class SyncResult(NamedTuple):
    updated: int
    skipped: int
    errors: int


def _to_jst_datetime(race_date: date, t: time) -> datetime:
    return datetime.combine(race_date, t, tzinfo=JST)


async def sync_race_schedule(
    session: Session,
    scraper: OddsparkScraper,
    target_date: date,
    *,
    tracks: list[TrackCode] | None = None,
) -> SyncResult:
    """Fetch start times from oddspark and update races in DB.

    For each race that already exists in the DB, sets:
    - scheduled_start_at
    - telvote_close_at (= scheduled_start_at - 2 minutes)
    - schedule_source
    - schedule_fetched_at
    """
    log = structlog.get_logger().bind(component="schedule_sync", target_date=str(target_date))
    target_tracks = tracks or list(TrackCode)
    now = datetime.now(tz=JST)
    updated = 0
    skipped = 0
    errors = 0

    for track in target_tracks:
        try:
            start_times = await scraper.get_all_race_start_times(track, target_date)
        except Exception as exc:
            log.warning("start_time_fetch_fail", track=track.name, error=str(exc))
            errors += 1
            continue

        if not start_times:
            log.debug("no_start_times", track=track.name)
            skipped += 1
            continue

        for race_no, start_time in start_times.items():
            race = session.execute(
                select(Race).where(
                    Race.track_id == track.value,
                    Race.race_date == target_date,
                    Race.race_number == race_no,
                )
            ).scalar_one_or_none()

            if race is None:
                log.debug("race_not_found", track=track.name, race_no=race_no)
                skipped += 1
                continue

            scheduled_start = _to_jst_datetime(target_date, start_time)
            telvote_close = scheduled_start - TELVOTE_OFFSET

            race.scheduled_start_at = scheduled_start.replace(tzinfo=None)
            race.telvote_close_at = telvote_close.replace(tzinfo=None)
            race.schedule_source = "oddspark"
            race.schedule_fetched_at = now.replace(tzinfo=None)
            updated += 1

        log.info(
            "track_synced",
            track=track.name,
            races_with_times=len(start_times),
        )

    if updated > 0:
        session.commit()

    log.info("sync_complete", updated=updated, skipped=skipped, errors=errors)
    return SyncResult(updated=updated, skipped=skipped, errors=errors)
