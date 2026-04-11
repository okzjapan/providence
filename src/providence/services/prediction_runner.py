"""Prediction runner service: core prediction logic decoupled from CLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime

import polars as pl
import structlog
from sqlalchemy.orm import Session

from providence.config import Settings, get_settings
from providence.database.repository import Repository
from providence.database.tables import Prediction, StrategyRun
from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.model.store import ModelStore
from providence.scraper.autorace_jp import AutoraceJpScraper
from providence.strategy.normalize import format_combination, market_odds_from_rows
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    RacePredictionBundle,
    StrategyConfig,
    StrategyRunResult,
)


@dataclass
class PredictionResult:
    """Result of a single race prediction, independent of display layer."""

    race_id: int
    track_code: TrackCode
    race_number: int
    target_date: date
    bundle: RacePredictionBundle
    strategy: StrategyRunResult
    program_sync_warning: str | None
    missing_trial_positions: list[int]
    has_market_odds: bool
    saved: bool = False


def resolve_judgment_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(tzinfo=None)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def build_prediction_rows(
    race_id: int,
    model_version: str,
    decision_time: datetime,
    strategy: StrategyRunResult,
) -> list[Prediction]:
    selected_lookup = {
        (bet.ticket_type, bet.combination): bet
        for bet in strategy.recommended_bets
    }
    rows = []
    for bet in (strategy.candidate_bets or strategy.recommended_bets):
        selected = selected_lookup.get((bet.ticket_type, bet.combination))
        rows.append(
            Prediction(
                race_id=race_id,
                model_version=model_version,
                predicted_at=decision_time,
                ticket_type=bet.ticket_type.value,
                combination=format_combination(bet.ticket_type, bet.combination),
                predicted_prob=bet.probability,
                odds_at_prediction=bet.odds_value,
                expected_value=bet.expected_value,
                kelly_fraction=bet.kelly_fraction,
                stake_weight=bet.stake_weight,
                recommended_bet=selected.recommended_bet if selected is not None else 0.0,
                confidence_score=bet.confidence_score,
                skip_reason=bet.skip_reason if selected is not None else strategy.skip_reason,
            )
        )
    return rows


def get_missing_trial_positions(race_df: pl.DataFrame) -> list[int]:
    if "trial_time" not in race_df.columns:
        return []
    missing_df = race_df.filter(pl.col("trial_time").is_null()).sort("post_position")
    return [int(value) for value in missing_df["post_position"].to_list()]


async def refresh_race_entries(
    target_date: date,
    track_code: TrackCode,
    race_no: int,
    repo: Repository,
    session_factory,
    *,
    settings: Settings | None = None,
) -> str | None:
    """Sync latest Program from autorace.jp. Returns warning string on failure, None on success."""
    settings = settings or get_settings()
    try:
        async with AutoraceJpScraper(settings) as scraper:
            entries = await scraper.get_race_entries(track_code, target_date, race_no)
    except Exception as exc:  # noqa: BLE001
        return f"Warning: 最新 Program の取得に失敗したため、既存 DB データで予測します。({exc})"

    if not entries.entries:
        return "Warning: 最新 Program に出走情報が無いため、既存 DB データで予測します。"

    try:
        with session_factory() as session:
            repo.save_race_data(session, entries, None, update_race_metadata=False)
    except Exception as exc:  # noqa: BLE001
        return f"Warning: 最新 Program の保存に失敗したため、既存 DB データで予測します。({exc})"
    return None


async def refresh_race_conditions(
    target_date: date,
    track_code: TrackCode,
    race_no: int,
    repo: Repository,
    session_factory,
    *,
    settings: Settings | None = None,
) -> str | None:
    """Fetch real-time weather/track conditions from Today API and update DB.

    Returns warning string on failure, None on success.
    """
    log = structlog.get_logger().bind(component="refresh_conditions")
    settings = settings or get_settings()
    try:
        async with AutoraceJpScraper(settings) as scraper:
            conditions = await scraper.get_live_conditions(track_code)
    except Exception as exc:  # noqa: BLE001
        log.debug("live_conditions_fetch_fail", error=str(exc))
        return f"Warning: リアルタイム走路状態の取得に失敗しました。DB の既存値で予測します。({exc})"

    if not conditions or not any(v is not None for v in conditions.values()):
        log.debug("live_conditions_empty", track=track_code.name)
        return None

    try:
        with session_factory() as session:
            updated = repo.update_race_conditions(
                session, track_code.value, target_date, conditions,
                force=True, race_number=race_no,
            )
        if updated:
            log.info(
                "live_conditions_updated",
                weather=conditions.get("weather"),
                track_condition=str(conditions.get("track_condition")),
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("live_conditions_save_fail", error=str(exc))
        return f"Warning: 走路状態の DB 更新に失敗しました。({exc})"

    return None


def run_prediction(
    *,
    target_date: date,
    track_code: TrackCode,
    race_number: int,
    decision_time: datetime,
    session_factory,
    repo: Repository,
    predictor: Predictor,
    loader: DataLoader,
    config: StrategyConfig | None = None,
    provenance: str = "service",
    skip_refresh: bool = False,
) -> PredictionResult:
    """Execute prediction for a single race. No I/O display, no save."""
    log = structlog.get_logger().bind(
        component="prediction_runner",
        track=track_code.name,
        race=race_number,
    )

    program_sync_warning = None
    if not skip_refresh:
        program_sync_warning = asyncio.run(
            refresh_race_entries(target_date, track_code, race_number, repo, session_factory)
        )

        conditions_warning = asyncio.run(
            refresh_race_conditions(target_date, track_code, race_number, repo, session_factory)
        )
        if conditions_warning and program_sync_warning is None:
            program_sync_warning = conditions_warning

    race_df = loader.load_race_dataset(start_date=target_date, end_date=target_date).filter(
        (pl.col("track_id") == track_code.value) & (pl.col("race_number") == race_number)
    )
    if race_df.is_empty():
        raise ValueError(
            f"対象レースのデータがありません: {track_code.japanese_name} {target_date} R{race_number}"
        )

    missing_trial = get_missing_trial_positions(race_df)
    if missing_trial:
        log.debug("missing_trial_positions", positions=missing_trial)

    predictor.load_history(target_date)
    bundle = predictor.predict_race(race_df)

    with session_factory() as session:
        odds_rows = repo.get_latest_market_odds(session, bundle.race_id, judgment_time=decision_time)

    market_odds = market_odds_from_rows(odds_rows)
    strategy = run_strategy(
        bundle,
        market_odds,
        decision_context=DecisionContext(
            judgment_time=decision_time,
            evaluation_mode=EvaluationMode.LIVE,
            timezone="UTC",
            provenance=provenance,
        ),
        config=config or StrategyConfig(),
    )

    return PredictionResult(
        race_id=bundle.race_id,
        track_code=track_code,
        race_number=race_number,
        target_date=target_date,
        bundle=bundle,
        strategy=strategy,
        program_sync_warning=program_sync_warning,
        missing_trial_positions=missing_trial,
        has_market_odds=bool(market_odds),
    )


def save_prediction(
    result: PredictionResult,
    decision_time: datetime,
    session_factory,
    repo: Repository,
    *,
    stake_sizing_rule: str = "min_100_normalized",
) -> None:
    """Persist a PredictionResult to DB as StrategyRun + Prediction rows."""
    with session_factory() as session:
        strategy_run = StrategyRun(
            race_id=result.bundle.race_id,
            model_version=result.bundle.model_version,
            evaluation_mode=EvaluationMode.LIVE.value,
            judgment_time=decision_time,
            stake_sizing_rule=stake_sizing_rule,
            confidence_score=result.strategy.confidence_score,
            skip_reason=result.strategy.skip_reason,
            total_recommended_bet=result.strategy.total_recommended_bet,
        )
        predictions = build_prediction_rows(
            result.bundle.race_id, result.bundle.model_version, decision_time, result.strategy
        )
        repo.save_strategy_run(session, strategy_run, predictions)
    result.saved = True
