"""Tests for Repository layer."""

from datetime import date

from providence.database.repository import Repository
from providence.database.tables import OddsSnapshot, Race, RaceEntry, RaceResult, Rider, Track
from providence.domain.enums import EntryStatus, Grade, TicketType, TrackCode
from providence.scraper.schemas import (
    EntryRow,
    PlayerSummary,
    RaceEntriesResponse,
    RaceResultResponse,
    RefundRow,
    ResultRow,
)


def _seed_tracks(session) -> None:
    repo = Repository()
    repo.ensure_tracks(session)


def _sample_entries_response() -> RaceEntriesResponse:
    return RaceEntriesResponse(
        track=TrackCode.HAMAMATSU,
        race_date=date(2025, 6, 15),
        race_number=1,
        grade=Grade.NORMAL,
        entries=[
            EntryRow(
                post_position=i,
                rider_registration_number=str(1000 + i),
                rider_name=f"Rider{i}",
                handicap_meters=i * 10,
            )
            for i in range(1, 9)
        ],
    )


def _sample_result_response() -> RaceResultResponse:
    return RaceResultResponse(
        track=TrackCode.HAMAMATSU,
        race_date=date(2025, 6, 15),
        race_number=1,
        results=[
            ResultRow(
                post_position=i,
                rider_registration_number=str(1000 + i),
                finish_position=i,
                race_time=3.3 + i * 0.01,
                entry_status=EntryStatus.RACING,
            )
            for i in range(1, 9)
        ],
        refunds=[
            RefundRow(ticket_type=TicketType.WIN, combination="1", refund_amount=350, popularity=1),
            RefundRow(ticket_type=TicketType.EXACTA, combination="1-2", refund_amount=1500, popularity=2),
        ],
    )


class TestEnsureTracks:
    def test_creates_5_tracks(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
        with session_factory() as session:
            tracks = session.query(Track).all()
            assert len(tracks) == 5
            names = {t.name for t in tracks}
            assert "浜松" in names

    def test_idempotent(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
        with session_factory() as session:
            repo.ensure_tracks(session)
        with session_factory() as session:
            assert session.query(Track).count() == 5


class TestSaveRaceData:
    def test_saves_entries_and_results(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        result_resp = _sample_result_response()

        with session_factory() as session:
            race = repo.save_race_data(session, entries_resp, result_resp)
            assert race.id is not None

        with session_factory() as session:
            assert session.query(Race).count() == 1
            assert session.query(Rider).count() == 8
            assert session.query(RaceEntry).count() == 8
            assert session.query(RaceResult).count() == 8
            assert session.query(OddsSnapshot).count() == 2

    def test_entries_only(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()

        with session_factory() as session:
            repo.save_race_data(session, entries_resp, None)

        with session_factory() as session:
            assert session.query(Race).count() == 1
            assert session.query(RaceEntry).count() == 8
            assert session.query(RaceResult).count() == 0

    def test_idempotent_upsert(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        result_resp = _sample_result_response()

        with session_factory() as session:
            repo.save_race_data(session, entries_resp, result_resp)
        with session_factory() as session:
            repo.save_race_data(session, entries_resp, result_resp)

        with session_factory() as session:
            assert session.query(Race).count() == 1
            assert session.query(Rider).count() == 8

    def test_refund_to_odds_conversion(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        result_resp = _sample_result_response()

        with session_factory() as session:
            repo.save_race_data(session, entries_resp, result_resp)

        with session_factory() as session:
            odds = session.query(OddsSnapshot).all()
            win_odds = [o for o in odds if o.ticket_type == TicketType.WIN.value]
            assert len(win_odds) == 1
            assert win_odds[0].odds_value == 3.5  # 350 / 100


class TestSavePlayer:
    def test_saves_player(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        player = PlayerSummary(
            registration_number="2913",
            name="テスト選手",
            age=35,
            generation=28,
            home_track=TrackCode.HAMAMATSU,
        )

        with session_factory() as session:
            repo.save_player(session, player)

        with session_factory() as session:
            rider = session.query(Rider).filter_by(registration_number="2913").one()
            assert rider.name == "テスト選手"
            assert rider.home_track_id == TrackCode.HAMAMATSU.value

    def test_upsert_updates_name(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        player1 = PlayerSummary(registration_number="2913", name="旧名前")
        player2 = PlayerSummary(registration_number="2913", name="新名前")

        with session_factory() as session:
            repo.save_player(session, player1)
        with session_factory() as session:
            repo.save_player(session, player2)

        with session_factory() as session:
            rider = session.query(Rider).filter_by(registration_number="2913").one()
            assert rider.name == "新名前"
            assert session.query(Rider).count() == 1


class TestGetCollectedRaces:
    def test_returns_collected(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        with session_factory() as session:
            repo.save_race_data(session, entries_resp, None)

        with session_factory() as session:
            collected = repo.get_collected_races(session, date(2025, 1, 1), date(2025, 12, 31))
            assert (TrackCode.HAMAMATSU.value, date(2025, 6, 15), 1) in collected

    def test_empty_when_no_data(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        with session_factory() as session:
            collected = repo.get_collected_races(session, date(2025, 1, 1), date(2025, 12, 31))
            assert len(collected) == 0


class TestDbStats:
    def test_returns_counts(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        with session_factory() as session:
            stats = repo.get_db_stats(session)
            assert stats["tracks"] == 5
            assert stats["riders"] == 0
