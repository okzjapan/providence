from datetime import date, datetime

import polars as pl
import pytest

from providence.services.prediction_runner import build_prediction_rows, get_missing_trial_positions, refresh_race_entries
from providence.domain.enums import TicketType, TrackCode
from providence.scraper.schemas import EntryRow, RaceEntriesResponse
from providence.strategy.types import DecisionContext, EvaluationMode, RecommendedBet, StrategyRunResult


def test_build_prediction_rows_keeps_candidate_rows_for_skipped_strategy():
    candidate = RecommendedBet(
        ticket_type=TicketType.WIN,
        combination=(1,),
        probability=0.2,
        odds_value=5.0,
        expected_value=0.0,
        confidence_score=0.8,
        kelly_fraction=0.01,
        recommended_bet=0.0,
        stake_weight=0.01,
    )
    strategy = StrategyRunResult(
        race_id=1,
        model_version="v001",
        decision_context=DecisionContext(
            judgment_time=datetime(2025, 6, 15, 10, 0, 0),
            evaluation_mode=EvaluationMode.LIVE,
        ),
        confidence_score=0.8,
        candidate_bets=[candidate],
        recommended_bets=[],
        skip_reason="rounded_below_minimum",
    )

    rows = build_prediction_rows(1, "v001", datetime(2025, 6, 15, 10, 0, 0), strategy)
    assert len(rows) == 1
    assert rows[0].recommended_bet == 0.0
    assert rows[0].skip_reason == "rounded_below_minimum"


def test_get_missing_trial_positions_returns_missing_cars():
    race_df = pl.DataFrame(
        {
            "post_position": [1, 2, 3, 4],
            "trial_time": [3.40, None, 3.36, None],
        }
    )

    assert get_missing_trial_positions(race_df) == [2, 4]


@pytest.mark.asyncio
async def test_refresh_race_entries_returns_warning_on_fetch_failure(monkeypatch):
    class DummyScraper:
        def __init__(self, settings):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        async def get_race_entries(self, track_code, target_date, race_no):  # noqa: ARG002
            raise RuntimeError("boom")

    monkeypatch.setattr("providence.services.prediction_runner.AutoraceJpScraper", DummyScraper)
    warning = await refresh_race_entries(date(2026, 4, 3), TrackCode.SANYO, 7, None, None)  # type: ignore[arg-type]
    assert "最新 Program の取得に失敗" in warning


@pytest.mark.asyncio
async def test_refresh_race_entries_returns_warning_on_empty_entries(monkeypatch):
    class DummyScraper:
        def __init__(self, settings):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        async def get_race_entries(self, track_code, target_date, race_no):  # noqa: ARG002
            return RaceEntriesResponse(track=track_code, race_date=target_date, race_number=race_no, entries=[])

    monkeypatch.setattr("providence.services.prediction_runner.AutoraceJpScraper", DummyScraper)
    warning = await refresh_race_entries(date(2026, 4, 3), TrackCode.SANYO, 7, None, None)  # type: ignore[arg-type]
    assert "最新 Program に出走情報が無い" in warning
