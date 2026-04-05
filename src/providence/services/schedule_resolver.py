"""Schedule resolver: find races approaching their voting deadline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from providence.database.tables import Race

JST = timezone(timedelta(hours=9))


class RaceDeadlineCandidate(NamedTuple):
    race_id: int
    track_id: int
    race_number: int
    telvote_close_at: datetime
    scheduled_start_at: datetime
    minutes_until_close: float


def find_approaching_deadlines(
    session: Session,
    *,
    now: datetime | None = None,
    lead_minutes: float = 5.0,
) -> list[RaceDeadlineCandidate]:
    """Return races whose telvote_close_at is within lead_minutes from now.

    Only races with:
    - telvote_close_at set (schedule sync has run)
    - telvote_close_at in the future
    - telvote_close_at - now <= lead_minutes
    """
    if now is None:
        now = datetime.now(tz=JST).replace(tzinfo=None)

    cutoff = now + timedelta(minutes=lead_minutes)

    races = list(
        session.execute(
            select(Race)
            .where(
                Race.telvote_close_at.is_not(None),
                Race.telvote_close_at > now,
                Race.telvote_close_at <= cutoff,
            )
            .order_by(Race.telvote_close_at, Race.track_id, Race.race_number)
        ).scalars()
    )

    candidates = []
    for race in races:
        minutes_left = (race.telvote_close_at - now).total_seconds() / 60.0
        candidates.append(
            RaceDeadlineCandidate(
                race_id=race.id,
                track_id=race.track_id,
                race_number=race.race_number,
                telvote_close_at=race.telvote_close_at,
                scheduled_start_at=race.scheduled_start_at,
                minutes_until_close=round(minutes_left, 1),
            )
        )

    log = structlog.get_logger().bind(component="schedule_resolver")
    log.info(
        "deadline_scan",
        now=now.isoformat(),
        lead_minutes=lead_minutes,
        candidates=len(candidates),
    )
    return candidates
