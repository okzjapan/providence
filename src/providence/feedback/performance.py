"""Performance aggregation and persistence for paper-trade operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session

from providence.database.repository import Repository
from providence.database.tables import BettingLog, Prediction, RaceEntry, RaceResult, StrategyRun
from providence.domain.enums import TicketType

WINDOWS: dict[str, int] = {"1w": 7, "4w": 28, "12w": 84}


@dataclass(frozen=True)
class PerformanceRefreshResult:
    rows_written: int
    computed_at: datetime


def refresh_model_performance(
    session: Session,
    *,
    end_date: date | None = None,
    repository: Repository | None = None,
) -> PerformanceRefreshResult:
    repository = repository or Repository()
    evaluation_date = end_date or date.today()
    computed_at = datetime.now(UTC).replace(tzinfo=None)
    rows: list[dict] = []
    for window, days in WINDOWS.items():
        start_date = evaluation_date - timedelta(days=days - 1)
        runs = repository.get_latest_strategy_runs(session, since_date=start_date)
        rows.extend(_build_rows_for_window(session, runs, evaluation_date, window, computed_at))
    written = repository.upsert_model_performance(session, rows)
    return PerformanceRefreshResult(rows_written=written, computed_at=computed_at)


def _build_rows_for_window(
    session: Session,
    runs: list[StrategyRun],
    evaluation_date: date,
    window: str,
    computed_at: datetime,
) -> list[dict]:
    if not runs:
        return []

    runs = [run for run in runs if run.judgment_time.date() <= evaluation_date]
    run_ids = [int(run.id) for run in runs]
    repo = Repository()
    predictions = repo.get_predictions_for_strategy_runs(session, run_ids)
    logs = repo.get_betting_logs_for_prediction_ids(session, [int(item.id) for item in predictions])
    logs_by_prediction = {int(item.prediction_id): item for item in logs}

    predictions_by_model: dict[str, list[tuple[Prediction, BettingLog | None]]] = {}
    for prediction in predictions:
        if prediction.strategy_run_id is None:
            continue
        pair = (prediction, logs_by_prediction.get(int(prediction.id)))
        predictions_by_model.setdefault(str(prediction.model_version), []).append(pair)

    rows: list[dict] = []
    for model_version, pairs in predictions_by_model.items():
        positive_pairs = [
            pair for pair in pairs if pair[1] is not None and float(pair[0].recommended_bet or 0.0) > 0
        ]
        win_pairs = [
            pair
            for pair in positive_pairs
            if pair[0].ticket_type == TicketType.WIN.value
        ]
        top3_accuracy = _top3_accuracy(session, win_pairs)
        roi = _roi(positive_pairs)
        win_accuracy = _win_accuracy(win_pairs)
        brier = _brier_score(win_pairs)
        calibration_error = _ece(win_pairs)
        rows.append(
            {
                "model_version": model_version,
                "evaluation_date": evaluation_date,
                "window": window,
                "sample_size": len(win_pairs),
                "win_accuracy": win_accuracy,
                "top3_accuracy": top3_accuracy,
                "brier_score": brier,
                "ndcg": None,
                "roi": roi,
                "calibration_error": calibration_error,
                "computed_at": computed_at,
            }
        )
    return rows


def _roi(pairs: list[tuple[Prediction, BettingLog | None]]) -> float | None:
    if not pairs:
        return None
    stake = sum(float(pred.recommended_bet or 0.0) for pred, _ in pairs)
    if stake <= 0:
        return None
    profit = sum(float(log.profit) for _, log in pairs if log is not None)
    return profit / stake


def _win_accuracy(pairs: list[tuple[Prediction, BettingLog | None]]) -> float | None:
    if not pairs:
        return None
    hits = sum(1 for _, log in pairs if log is not None and float(log.payout) > 0)
    return hits / len(pairs)


def _brier_score(pairs: list[tuple[Prediction, BettingLog | None]]) -> float | None:
    if not pairs:
        return None
    values = [
        (float(pred.predicted_prob), 1.0 if log is not None and float(log.payout) > 0 else 0.0)
        for pred, log in pairs
    ]
    return mean((prob - label) ** 2 for prob, label in values)


def _ece(pairs: list[tuple[Prediction, BettingLog | None]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    bucket_values: dict[int, list[tuple[float, float]]] = {}
    for pred, log in pairs:
        prob = float(pred.predicted_prob)
        label = 1.0 if log is not None and float(log.payout) > 0 else 0.0
        bucket = min(int(prob * bins), bins - 1)
        bucket_values.setdefault(bucket, []).append((prob, label))

    total = len(pairs)
    ece = 0.0
    for values in bucket_values.values():
        avg_prob = mean(prob for prob, _ in values)
        avg_label = mean(label for _, label in values)
        ece += (len(values) / total) * abs(avg_prob - avg_label)
    return ece


def _top3_accuracy(session: Session, pairs: list[tuple[Prediction, BettingLog | None]]) -> float | None:
    if not pairs:
        return None
    race_ids = sorted({int(prediction.race_id) for prediction, _ in pairs})
    results = list(
        session.execute(
            select(RaceEntry.post_position, RaceEntry.race_id, RaceResult.finish_position)
            .join(RaceResult, RaceResult.race_entry_id == RaceEntry.id)
            .where(RaceEntry.race_id.in_(race_ids))
        )
    )
    finish_map = {
        (int(race_id), int(post_position)): finish_position
        for post_position, race_id, finish_position in results
        if finish_position is not None
    }
    hits = 0
    total = 0
    for prediction, _ in pairs:
        try:
            post_position = int(prediction.combination)
        except ValueError:
            continue
        finish_position = finish_map.get((int(prediction.race_id), post_position))
        if finish_position is None:
            continue
        total += 1
        if int(finish_position) <= 3:
            hits += 1
    if total == 0:
        return None
    return hits / total
