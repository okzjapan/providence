"""Tests for schedule sync service."""

from datetime import date, datetime, time, timedelta

import pytest

from providence.database.repository import Repository
from providence.database.tables import Race
from providence.services.schedule_sync import SyncResult, _to_jst_datetime, sync_race_schedule, TELVOTE_OFFSET


class FakeOddsparkScraper:
    """Fake scraper returning predetermined start times."""

    def __init__(self, start_times: dict[int, dict[int, time]]):
        self._start_times = start_times

    async def get_all_race_start_times(self, track, race_date):
        return self._start_times.get(track.value, {})


@pytest.fixture()
def _seed_races(session_factory):
    repo = Repository()
    with session_factory() as session:
        repo.ensure_tracks(session)
        with session.begin():
            for race_no in range(1, 13):
                session.add(Race(
                    track_id=3,
                    race_date=date(2026, 4, 4),
                    race_number=race_no,
                    grade="普通",
                    distance=3100,
                ))


@pytest.mark.usefixtures("_seed_races")
class TestScheduleSync:
    @pytest.mark.asyncio
    async def test_sync_updates_races(self, session_factory):
        from providence.domain.enums import TrackCode

        fake = FakeOddsparkScraper({
            3: {
                1: time(10, 56),
                2: time(11, 21),
                12: time(16, 35),
            },
        })

        with session_factory() as session:
            result = await sync_race_schedule(
                session, fake, date(2026, 4, 4), tracks=[TrackCode.ISESAKI]
            )

        assert result.updated == 3
        assert result.errors == 0

        with session_factory() as session:
            r1 = session.query(Race).filter_by(track_id=3, race_number=1).one()
            assert r1.scheduled_start_at == datetime(2026, 4, 4, 10, 56)
            assert r1.telvote_close_at == datetime(2026, 4, 4, 10, 54)
            assert r1.schedule_source == "oddspark"
            assert r1.schedule_fetched_at is not None

            r12 = session.query(Race).filter_by(track_id=3, race_number=12).one()
            assert r12.scheduled_start_at == datetime(2026, 4, 4, 16, 35)
            assert r12.telvote_close_at == datetime(2026, 4, 4, 16, 33)

    @pytest.mark.asyncio
    async def test_sync_skips_missing_races(self, session_factory):
        from providence.domain.enums import TrackCode

        fake = FakeOddsparkScraper({2: {1: time(10, 0)}})

        with session_factory() as session:
            result = await sync_race_schedule(
                session, fake, date(2026, 4, 4), tracks=[TrackCode.KAWAGUCHI]
            )

        assert result.updated == 0
        assert result.skipped >= 1

    @pytest.mark.asyncio
    async def test_sync_handles_empty_response(self, session_factory):
        from providence.domain.enums import TrackCode

        fake = FakeOddsparkScraper({})

        with session_factory() as session:
            result = await sync_race_schedule(
                session, fake, date(2026, 4, 4), tracks=[TrackCode.ISESAKI]
            )

        assert result.updated == 0


class TestToJstDatetime:
    def test_combines_date_and_time(self):
        result = _to_jst_datetime(date(2026, 4, 4), time(16, 35))
        assert result.hour == 16
        assert result.minute == 35
        assert result.tzinfo is not None


class TestTelvoteOffset:
    def test_offset_is_two_minutes(self):
        assert TELVOTE_OFFSET == timedelta(minutes=2)
