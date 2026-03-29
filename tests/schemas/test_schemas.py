"""Tests for Pydantic schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from providence.domain.enums import EntryStatus, Grade, TicketType, TrackCode
from providence.scraper.schemas import (
    EntryRow,
    OddsRow,
    RaceEntriesResponse,
    RaceResultResponse,
    RefundRow,
    ResultRow,
)


class TestEntryRow:
    def test_valid(self):
        row = EntryRow(
            post_position=1,
            rider_registration_number="2913",
            rider_name="テスト選手",
            handicap_meters=10,
        )
        assert row.post_position == 1
        assert row.trial_time is None

    def test_with_all_fields(self):
        row = EntryRow(
            post_position=3,
            rider_registration_number="1234",
            rider_name="選手A",
            age=35,
            generation=28,
            handicap_meters=30,
            trial_time=3.35,
            avg_trial_time=3.40,
            trial_deviation=0.05,
            race_score=55.5,
        )
        assert row.trial_time == 3.35

    def test_post_position_out_of_range(self):
        with pytest.raises(ValidationError):
            EntryRow(
                post_position=0,
                rider_registration_number="1234",
                rider_name="X",
                handicap_meters=10,
            )

    def test_negative_handicap(self):
        with pytest.raises(ValidationError):
            EntryRow(
                post_position=1,
                rider_registration_number="1234",
                rider_name="X",
                handicap_meters=-10,
            )


class TestRaceEntriesResponse:
    def test_valid(self):
        resp = RaceEntriesResponse(
            track=TrackCode.HAMAMATSU,
            race_date=date(2026, 3, 29),
            race_number=1,
            entries=[
                EntryRow(
                    post_position=i, rider_registration_number=str(1000 + i),
                    rider_name=f"R{i}", handicap_meters=0,
                )
                for i in range(1, 9)
            ],
        )
        assert len(resp.entries) == 8
        assert resp.grade == Grade.NORMAL

    def test_race_number_out_of_range(self):
        with pytest.raises(ValidationError):
            RaceEntriesResponse(
                track=TrackCode.KAWAGUCHI,
                race_date=date(2026, 1, 1),
                race_number=13,
                entries=[],
            )


class TestResultRow:
    def test_valid(self):
        row = ResultRow(
            post_position=1,
            rider_registration_number="2913",
            finish_position=1,
            race_time=3.321,
            entry_status=EntryStatus.RACING,
        )
        assert row.finish_position == 1

    def test_cancelled(self):
        row = ResultRow(
            post_position=2,
            rider_registration_number="3000",
            entry_status=EntryStatus.CANCELLED,
        )
        assert row.finish_position is None


class TestRefundRow:
    def test_valid(self):
        row = RefundRow(
            ticket_type=TicketType.EXACTA,
            combination="1-3",
            refund_amount=2850,
            popularity=3,
        )
        assert row.refund_amount == 2850

    def test_negative_refund(self):
        with pytest.raises(ValidationError):
            RefundRow(
                ticket_type=TicketType.WIN,
                combination="1",
                refund_amount=-100,
            )


class TestOddsRow:
    def test_valid(self):
        row = OddsRow(
            ticket_type=TicketType.WIN,
            combination="1",
            odds_value=2.5,
        )
        assert row.odds_value == 2.5

    def test_zero_odds_rejected(self):
        with pytest.raises(ValidationError):
            OddsRow(
                ticket_type=TicketType.WIN,
                combination="1",
                odds_value=0.0,
            )


class TestRaceResultResponse:
    def test_valid(self):
        resp = RaceResultResponse(
            track=TrackCode.IIZUKA,
            race_date=date(2025, 6, 15),
            race_number=8,
            results=[
                ResultRow(post_position=1, rider_registration_number="1001", finish_position=1),
            ],
            refunds=[
                RefundRow(ticket_type=TicketType.WIN, combination="1", refund_amount=350),
            ],
        )
        assert len(resp.results) == 1
        assert resp.refunds[0].refund_amount == 350
