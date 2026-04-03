from datetime import date, datetime

from providence.database.repository import Repository
from providence.database.tables import Prediction, Race, StrategyRun, TicketPayout, Track
from providence.feedback.reconcile import reconcile_paper_trades


def test_reconcile_creates_betting_logs_idempotently(session_factory):
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
            race_id=1,
            strategy_run_id=run.id,
            model_version="v003",
            predicted_at=datetime(2026, 4, 2, 10, 0, 0),
            ticket_type="単勝",
            combination="1",
            predicted_prob=0.2,
            recommended_bet=100.0,
        )
        session.add(prediction)
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
        result = reconcile_paper_trades(session, repository=repo)
        assert result.betting_logs_written == 1

    with session_factory() as session:
        result = reconcile_paper_trades(session, repository=repo)
        logs = repo.get_betting_logs_for_prediction_ids(session, [1])
        assert result.betting_logs_written == 1
        assert len(logs) == 1
        assert logs[0].payout == 350.0
        assert logs[0].profit == 250.0
