"""Paper-trade reconciliation between predictions and settled payouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from providence.database.repository import Repository
from providence.database.tables import Prediction, StrategyRun, TicketPayout


@dataclass(frozen=True)
class ReconcileResult:
    strategy_runs: int
    predictions_seen: int
    payouts_missing_races: int
    betting_logs_written: int
    reconciled_at: datetime


def reconcile_paper_trades(
    session: Session,
    *,
    repository: Repository | None = None,
    since_date: date | None = None,
) -> ReconcileResult:
    repository = repository or Repository()
    strategy_runs = repository.get_latest_strategy_runs(session, since_date=since_date)
    run_ids = [run.id for run in strategy_runs]
    predictions = repository.get_predictions_for_strategy_runs(session, run_ids)
    predictions_by_run: dict[int, list[Prediction]] = {}
    for prediction in predictions:
        if prediction.strategy_run_id is None:
            continue
        predictions_by_run.setdefault(int(prediction.strategy_run_id), []).append(prediction)

    race_ids = sorted({int(run.race_id) for run in strategy_runs})
    payouts_by_race = repository.get_ticket_payouts_for_races(session, race_ids)
    reconciled_at = datetime.now(UTC).replace(tzinfo=None)

    rows: list[dict] = []
    payouts_missing_races = 0
    for run in strategy_runs:
        run_predictions = predictions_by_run.get(int(run.id), [])
        race_payouts = payouts_by_race.get(int(run.race_id), [])
        if not race_payouts:
            payouts_missing_races += 1
            continue
        payout_lookup = {
            (p.ticket_type, p.combination): p
            for p in race_payouts
        }
        for prediction in run_predictions:
            actual_bet = float(prediction.recommended_bet or 0.0)
            payout_row = payout_lookup.get((prediction.ticket_type, prediction.combination))
            payout_value = float(payout_row.payout_value) if payout_row is not None else 0.0
            payout_amount = actual_bet * payout_value if actual_bet > 0 and payout_value > 0 else 0.0
            rows.append(
                {
                    "prediction_id": int(prediction.id),
                    "race_id": int(prediction.race_id),
                    "actual_bet_amount": actual_bet,
                    "payout": payout_amount,
                    "profit": payout_amount - actual_bet,
                    "bankroll_after": None,
                    "reconciled_at": reconciled_at,
                }
            )

    written = repository.upsert_betting_logs(session, rows)
    return ReconcileResult(
        strategy_runs=len(strategy_runs),
        predictions_seen=len(predictions),
        payouts_missing_races=payouts_missing_races,
        betting_logs_written=written,
        reconciled_at=reconciled_at,
    )
