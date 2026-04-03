"""Tests for Repository layer."""

from datetime import date, datetime

import pytest

from providence.database.repository import Repository
from providence.database.tables import OddsSnapshot, Race, RaceEntry, RaceResult, Rider, TicketPayout, Track
from providence.domain.enums import EntryStatus, Grade, TicketType, TrackCode
from providence.scraper.schemas import EntryRow, OddsRow, PlayerSummary, RaceEntriesResponse, RaceResultResponse, RefundRow, ResultRow


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
            assert session.query(TicketPayout).count() == 2

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

    def test_second_save_updates_trial_times(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        updated_entries = entries_resp.model_copy(
            update={
                "entries": [
                    entry.model_copy(update={"trial_time": 3.30 + idx * 0.01})
                    for idx, entry in enumerate(entries_resp.entries, start=1)
                ]
            }
        )

        with session_factory() as session:
            repo.save_race_data(session, entries_resp, None)
        with session_factory() as session:
            repo.save_race_data(session, updated_entries, None)

        with session_factory() as session:
            rows = session.query(RaceEntry).order_by(RaceEntry.post_position).all()
            assert rows[0].trial_time == pytest.approx(3.31)
            assert rows[-1].trial_time == pytest.approx(3.38)

    def test_refund_to_ticket_payout_conversion(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)

        entries_resp = _sample_entries_response()
        result_resp = _sample_result_response()

        with session_factory() as session:
            repo.save_race_data(session, entries_resp, result_resp)

        with session_factory() as session:
            payouts = session.query(TicketPayout).all()
            win_payouts = [o for o in payouts if o.ticket_type == TicketType.WIN.value]
            assert len(win_payouts) == 1
            assert win_payouts[0].payout_value == 3.5  # 350 / 100

    def test_save_odds_assigns_batch_id_and_get_latest_market_batch(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
            race = repo.save_race_data(session, _sample_entries_response(), None)

        with session_factory() as session:
            repo.save_odds(
                session,
                race.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="1", odds_value=2.1)],
                captured_at=datetime(2025, 6, 15, 10, 0, 0),
                ingestion_batch_id="batch-1",
            )
        with session_factory() as session:
            repo.save_odds(
                session,
                race.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="1", odds_value=2.8)],
                captured_at=datetime(2025, 6, 15, 11, 0, 0),
                ingestion_batch_id="batch-2",
            )

        with session_factory() as session:
            latest = repo.get_latest_market_odds(session, race.id, judgment_time=datetime(2025, 6, 15, 10, 30, 0))
            assert len(latest) == 1
            assert latest[0].ingestion_batch_id == "batch-1"
            assert latest[0].odds_value == 2.1

    def test_save_odds_is_safe_after_read_in_same_session(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
            repo.save_race_data(session, _sample_entries_response(), None)

        with session_factory() as session:
            race = repo.get_race(session, TrackCode.HAMAMATSU.value, date(2025, 6, 15), 1)
            assert race is not None
            inserted = repo.save_odds(
                session,
                race.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="1", odds_value=2.5)],
                ingestion_batch_id="batch-read-safe",
            )
            assert inserted == 1

    def test_legacy_odds_without_batch_are_ignored(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
            race = repo.save_race_data(session, _sample_entries_response(), None)
            session.add(
                OddsSnapshot(
                    race_id=race.id,
                    ticket_type=TicketType.WIN.value,
                    combination="1",
                    odds_value=9.9,
                    captured_at=datetime(2025, 6, 15, 12, 0, 0),
                )
            )
            session.commit()

        with session_factory() as session:
            latest = repo.get_latest_market_odds(session, race.id, judgment_time=datetime(2025, 6, 15, 13, 0, 0))
            assert latest == []

    def test_get_latest_market_odds_for_races_returns_batch_per_race(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
            race1 = repo.save_race_data(session, _sample_entries_response(), None)
            race2_entries = _sample_entries_response().model_copy(update={"race_number": 2})
            race2 = repo.save_race_data(session, race2_entries, None)

        with session_factory() as session:
            repo.save_odds(
                session,
                race1.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="1", odds_value=2.1)],
                captured_at=datetime(2025, 6, 15, 10, 0, 0),
                ingestion_batch_id="r1-old",
            )
        with session_factory() as session:
            repo.save_odds(
                session,
                race1.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="1", odds_value=2.5)],
                captured_at=datetime(2025, 6, 15, 11, 0, 0),
                ingestion_batch_id="r1-new",
            )
        with session_factory() as session:
            repo.save_odds(
                session,
                race2.id,
                [OddsRow(ticket_type=TicketType.WIN, combination="2", odds_value=4.0)],
                captured_at=datetime(2025, 6, 15, 10, 30, 0),
                ingestion_batch_id="r2-only",
            )

        with session_factory() as session:
            grouped = repo.get_latest_market_odds_for_races(
                session,
                [race1.id, race2.id],
                judgment_time=datetime(2025, 6, 15, 10, 45, 0),
            )
            assert grouped[race1.id][0].ingestion_batch_id == "r1-old"
            assert grouped[race2.id][0].ingestion_batch_id == "r2-only"

    def test_backfill_legacy_payouts_is_idempotent(self, session_factory):
        repo = Repository()
        with session_factory() as session:
            repo.ensure_tracks(session)
            race = repo.save_race_data(session, _sample_entries_response(), None)
            session.add(
                OddsSnapshot(
                    race_id=race.id,
                    ticket_type=TicketType.WIN.value,
                    combination="1",
                    odds_value=3.5,
                    captured_at=datetime(2025, 6, 15, 12, 0, 0),
                )
            )
            session.commit()

        with session_factory() as session:
            inserted = repo.backfill_legacy_payouts(session)
            assert inserted == 1

        with session_factory() as session:
            inserted = repo.backfill_legacy_payouts(session)
            payouts = session.query(TicketPayout).all()
            assert inserted == 0
            assert len(payouts) == 1
            assert payouts[0].payout_value == 3.5


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
