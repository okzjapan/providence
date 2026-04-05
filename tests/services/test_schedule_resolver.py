"""Tests for schedule resolver service."""

from datetime import date, datetime, timedelta

import pytest

from providence.database.repository import Repository
from providence.database.tables import Race
from providence.services.schedule_resolver import RaceDeadlineCandidate, find_approaching_deadlines


@pytest.fixture()
def _seed_races_with_schedule(session_factory):
    repo = Repository()
    with session_factory() as session:
        repo.ensure_tracks(session)
        with session.begin():
            session.add(Race(
                track_id=3,
                race_date=date(2026, 4, 4),
                race_number=3,
                grade="普通",
                distance=3100,
                scheduled_start_at=datetime(2026, 4, 4, 11, 47),
                telvote_close_at=datetime(2026, 4, 4, 11, 45),
                schedule_source="oddspark",
                schedule_fetched_at=datetime(2026, 4, 4, 7, 0),
            ))
            session.add(Race(
                track_id=3,
                race_date=date(2026, 4, 4),
                race_number=4,
                grade="普通",
                distance=3100,
                scheduled_start_at=datetime(2026, 4, 4, 12, 15),
                telvote_close_at=datetime(2026, 4, 4, 12, 13),
                schedule_source="oddspark",
                schedule_fetched_at=datetime(2026, 4, 4, 7, 0),
            ))
            session.add(Race(
                track_id=3,
                race_date=date(2026, 4, 4),
                race_number=12,
                grade="普通",
                distance=3100,
                scheduled_start_at=datetime(2026, 4, 4, 16, 35),
                telvote_close_at=datetime(2026, 4, 4, 16, 33),
                schedule_source="oddspark",
                schedule_fetched_at=datetime(2026, 4, 4, 7, 0),
            ))
            # Race without schedule (should be excluded)
            session.add(Race(
                track_id=3,
                race_date=date(2026, 4, 4),
                race_number=1,
                grade="普通",
                distance=3100,
            ))


@pytest.mark.usefixtures("_seed_races_with_schedule")
class TestFindApproachingDeadlines:
    def test_finds_race_within_window(self, session_factory):
        now = datetime(2026, 4, 4, 11, 41)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=5.0)

        assert len(candidates) == 1
        assert candidates[0].race_number == 3
        assert candidates[0].minutes_until_close == 4.0

    def test_excludes_past_deadlines(self, session_factory):
        now = datetime(2026, 4, 4, 11, 46)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=5.0)

        assert not any(c.race_number == 3 for c in candidates)

    def test_excludes_far_future(self, session_factory):
        now = datetime(2026, 4, 4, 11, 0)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=5.0)

        assert len(candidates) == 0

    def test_multiple_races_in_window(self, session_factory):
        now = datetime(2026, 4, 4, 12, 9)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=5.0)

        assert len(candidates) == 1
        assert candidates[0].race_number == 4

    def test_excludes_race_without_schedule(self, session_factory):
        now = datetime(2026, 4, 4, 10, 0)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=1440.0)

        race_numbers = [c.race_number for c in candidates]
        assert 1 not in race_numbers

    def test_returns_sorted_by_deadline(self, session_factory):
        now = datetime(2026, 4, 4, 11, 40)

        with session_factory() as session:
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=60.0)

        deadlines = [c.telvote_close_at for c in candidates]
        assert deadlines == sorted(deadlines)
