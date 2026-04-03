from datetime import date, datetime

from providence.database.repository import Repository
from providence.database.tables import (
    BettingLog,
    Prediction,
    Race,
    StrategyRun,
    TicketPayout,
    Track,
)
from providence.feedback.performance import refresh_model_performance


def test_refresh_model_performance_upserts_rows(session_factory):
    repo = Repository()
    with session_factory() as session:
        session.add(Track(id=1, name="川口", location="埼玉"))
        session.add(Race(id=1, track_id=1, race_date=date(2026, 4, 2), race_number=1, grade="普通", distance=3100))
        session.flush()
        run = StrategyRun(
            race_id=1,
            model_version="v003",
            evaluation_mode="live",
            judgment_time=datetime(2026, 4, 2, 10, 0, 0),
            total_recommended_bet=100.0,
        )
        session.add(run)
        session.flush()
        prediction = Prediction(
            id=1,
            race_id=1,
            strategy_run_id=run.id,
            model_version="v003",
            predicted_at=datetime(2026, 4, 2, 10, 0, 0),
            ticket_type="単勝",
            combination="1",
            predicted_prob=0.3,
            recommended_bet=100.0,
        )
        session.add(prediction)
        session.add(
            BettingLog(
                prediction_id=1,
                race_id=1,
                actual_bet_amount=100.0,
                payout=350.0,
                profit=250.0,
                reconciled_at=datetime(2026, 4, 2, 21, 0, 0),
            )
        )
        session.add(
            TicketPayout(
                race_id=1,
                ticket_type="単勝",
                combination="1",
                payout_value=3.5,
                settled_at=datetime(2026, 4, 2, 20, 0, 0),
            )
        )
        session.commit()

    with session_factory() as session:
        result = refresh_model_performance(session, repository=repo, end_date=date(2026, 4, 2))
        assert result.rows_written == 3

    with session_factory() as session:
        result = refresh_model_performance(session, repository=repo, end_date=date(2026, 4, 2))
        rows = repo.get_recent_model_performance(session, model_version="v003", limit=10)
        assert result.rows_written == 3
        assert len(rows) == 3
