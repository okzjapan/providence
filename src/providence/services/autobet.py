"""Autobet orchestration: 2-phase prediction with Slack notification.

Phase 1 (T-12min to T-7min):
  - Scrape entries/conditions
  - Run predict (no odds) → confidence is determined here
  - If low confidence → Slack "非推奨" immediately (with win probs)
  - If high confidence → cache prediction bundle for Phase 2

Phase 2 (T-7min to T-0min):
  - Scrape LATEST odds (and save to DB)
  - Reload strategy with cached bundle + fresh odds → Slack notify
"""

from __future__ import annotations

import asyncio
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from providence.config import Settings, get_settings
from providence.database.repository import Repository
from providence.domain.enums import TrackCode
from providence.features.loader import DataLoader
from providence.features.pipeline import FeaturePipeline
from providence.model.predictor import Predictor
from providence.notification.slack import send_prediction_to_slack
from providence.scraper.autorace_jp import AutoraceJpScraper
from providence.scraper.oddspark import OddsparkScraper
from providence.services.prediction_runner import (
    PredictionResult,
    resolve_judgment_time,
    run_prediction,
    save_prediction,
)
from providence.services.schedule_resolver import RaceDeadlineCandidate, find_approaching_deadlines
from providence.services.schedule_sync import sync_race_schedule
from providence.strategy.normalize import market_odds_from_rows
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    RacePredictionBundle,
    StrategyConfig,
)

JST = timezone(timedelta(hours=9))
PHASE2_THRESHOLD_MINUTES = 7.0
HISTORY_CACHE_DIR = Path("data/cache")
BUNDLE_CACHE_DIR = Path("data/cache/bundles")

log = structlog.get_logger()


@dataclass
class TickResult:
    """Single race processing result within a tick."""

    race_id: int
    track_name: str
    race_number: int
    status: str
    phase: str = ""
    has_recommended_bets: bool = False
    total_recommended_bet: float = 0.0
    slack_sent: bool = False
    error: str | None = None


def run_autobet_tick(
    *,
    session_factory,
    repo: Repository,
    predictor: Predictor,
    loader: DataLoader,
    config: StrategyConfig | None = None,
    settings: Settings | None = None,
    slack_webhook_url: str | None = None,
    slack_mention: str = "",
    lead_minutes: float = 10.0,
    save: bool = True,
    dry_run: bool = False,
    force_race: tuple[TrackCode, int] | None = None,
) -> list[TickResult]:
    """3分ごとに呼ばれる tick（同期関数）。"""
    settings = settings or get_settings()
    config = config or StrategyConfig()

    with session_factory() as session:
        if force_race:
            candidates = _build_forced_candidates(session, repo, force_race[0], force_race[1])
        else:
            now = datetime.now(tz=JST).replace(tzinfo=None)
            candidates = find_approaching_deadlines(session, now=now, lead_minutes=lead_minutes)

    if not candidates:
        log.debug("autobet_tick_no_candidates", lead_minutes=lead_minutes)
        return []

    log.info("autobet_tick_start", candidates=len(candidates))
    results = []
    for candidate in candidates:
        phase = _determine_phase(candidate, session_factory, repo, force_race is not None)
        if phase == "skip":
            continue

        if phase == "phase1":
            result = _run_phase1(
                candidate=candidate,
                session_factory=session_factory,
                repo=repo,
                predictor=predictor,
                loader=loader,
                config=config,
                settings=settings,
                slack_webhook_url=slack_webhook_url,
                dry_run=dry_run,
            )
        else:
            result = _run_phase2(
                candidate=candidate,
                session_factory=session_factory,
                repo=repo,
                predictor=predictor,
                loader=loader,
                config=config,
                settings=settings,
                slack_webhook_url=slack_webhook_url,
                slack_mention=slack_mention,
                save=save,
                dry_run=dry_run,
            )
        results.append(result)

    if results:
        p1 = sum(1 for r in results if r.phase == "phase1")
        p2 = sum(1 for r in results if r.phase == "phase2")
        bets = sum(1 for r in results if r.has_recommended_bets)
        log.info("autobet_tick_done", phase1=p1, phase2=p2, with_bets=bets)
    return results


def _determine_phase(
    candidate: RaceDeadlineCandidate,
    session_factory,
    repo: Repository,
    is_forced: bool,
) -> str:
    if is_forced:
        return "phase2"

    with session_factory() as session:
        if _phase2_already_done(session, candidate.race_id):
            return "skip"

    if _phase1_already_done(candidate.race_id):
        if candidate.minutes_until_close > PHASE2_THRESHOLD_MINUTES:
            return "skip"
        return "phase2"

    if candidate.minutes_until_close > PHASE2_THRESHOLD_MINUTES:
        return "phase1"
    return "phase2"


# ---------------------------------------------------------------------------
# Phase 1: Predict without odds → notify if low confidence
# ---------------------------------------------------------------------------

def _run_phase1(
    *,
    candidate: RaceDeadlineCandidate,
    session_factory,
    repo: Repository,
    predictor: Predictor,
    loader: DataLoader,
    config: StrategyConfig,
    settings: Settings,
    slack_webhook_url: str | None,
    dry_run: bool,
) -> TickResult:
    """Phase 1: 予測実行（オッズなし）。低信頼度なら即 Slack 通知。"""
    track_code = TrackCode(candidate.track_id)
    track_name = track_code.japanese_name
    race_number = candidate.race_number
    log_ctx = log.bind(track=track_name, race=race_number, phase="phase1")

    target_date = _extract_date(candidate)

    try:
        _scrape_entries_and_conditions(track_code, target_date, race_number, session_factory, repo, settings)
    except Exception as exc:  # noqa: BLE001
        log_ctx.warning("phase1_scrape_failed", error=str(exc))

    _load_history_with_cache(predictor, loader, target_date)

    decision_time = resolve_judgment_time(None)
    try:
        prediction_result = run_prediction(
            target_date=target_date,
            track_code=track_code,
            race_number=race_number,
            decision_time=decision_time,
            session_factory=session_factory,
            repo=repo,
            predictor=predictor,
            loader=loader,
            config=config,
            provenance="autobet.phase1",
            skip_refresh=True,
        )
    except Exception as exc:  # noqa: BLE001
        log_ctx.error("phase1_prediction_failed", error=str(exc))
        return TickResult(
            race_id=candidate.race_id, track_name=track_name,
            race_number=race_number, status="prediction_failed", phase="phase1", error=str(exc),
        )

    confidence = prediction_result.strategy.confidence_score
    is_low_confidence = confidence < config.min_confidence

    if is_low_confidence:
        log_ctx.info("phase1_low_confidence", confidence=round(confidence, 4))
        slack_sent = False
        if slack_webhook_url and not dry_run:
            slack_sent = send_prediction_to_slack(slack_webhook_url, prediction_result)
        elif dry_run:
            _log_dry_run(prediction_result)
        _mark_phase1_done(candidate.race_id, notify_sent=True)
        return TickResult(
            race_id=candidate.race_id, track_name=track_name,
            race_number=race_number, status="phase1_notified_low_conf",
            phase="phase1", slack_sent=slack_sent,
        )

    slack_sent = False
    if _was_previously_non_candidate(candidate.race_id):
        log_ctx.info("phase1_upgraded_to_candidate", confidence=round(confidence, 4))
        if slack_webhook_url and not dry_run:
            from providence.notification.slack import SLACK_MENTION_OKZ

            upgrade_text = (
                f"{SLACK_MENTION_OKZ} 📈 *{track_name} R{race_number}* が投票候補に昇格しました"
                f"（試走タイム反映後の信頼度: *{confidence:.2f}*）\nPhase 2 で最終判定します"
            )
            import httpx

            try:
                httpx.post(slack_webhook_url, json={"text": upgrade_text}, timeout=10.0)
                slack_sent = True
            except httpx.HTTPError:
                pass

    _save_bundle_cache(candidate.race_id, prediction_result.bundle)
    _mark_phase1_done(candidate.race_id, notify_sent=False)
    log_ctx.info("phase1_high_confidence", confidence=round(confidence, 4))

    return TickResult(
        race_id=candidate.race_id, track_name=track_name,
        race_number=race_number, status="phase1_cached", phase="phase1",
    )


# ---------------------------------------------------------------------------
# Phase 2: Fresh odds → strategy → Slack notify
# ---------------------------------------------------------------------------

def _run_phase2(
    *,
    candidate: RaceDeadlineCandidate,
    session_factory,
    repo: Repository,
    predictor: Predictor,
    loader: DataLoader,
    config: StrategyConfig,
    settings: Settings,
    slack_webhook_url: str | None,
    slack_mention: str,
    save: bool,
    dry_run: bool,
) -> TickResult:
    """Phase 2: 最新オッズ → 戦略 → Slack 通知。"""
    track_code = TrackCode(candidate.track_id)
    track_name = track_code.japanese_name
    race_number = candidate.race_number
    log_ctx = log.bind(track=track_name, race=race_number, phase="phase2")

    target_date = _extract_date(candidate)
    decision_time = resolve_judgment_time(None)

    try:
        _scrape_odds_for_race(track_code, target_date, race_number, session_factory, repo, settings)
        log_ctx.info("phase2_odds_scraped")
    except Exception as exc:  # noqa: BLE001
        log_ctx.warning("phase2_odds_scrape_failed", error=str(exc))

    cached_bundle = _load_bundle_cache(candidate.race_id)

    if cached_bundle is not None:
        prediction_result = _rebuild_prediction_from_cache(
            cached_bundle, candidate, session_factory, repo, config, decision_time,
        )
        log_ctx.info("phase2_used_cached_bundle")
    else:
        _load_history_with_cache(predictor, loader, target_date)
        try:
            prediction_result = run_prediction(
                target_date=target_date,
                track_code=track_code,
                race_number=race_number,
                decision_time=decision_time,
                session_factory=session_factory,
                repo=repo,
                predictor=predictor,
                loader=loader,
                config=config,
                provenance="autobet.phase2",
                skip_refresh=True,
            )
        except Exception as exc:  # noqa: BLE001
            log_ctx.error("phase2_prediction_failed", error=str(exc))
            return TickResult(
                race_id=candidate.race_id, track_name=track_name,
                race_number=race_number, status="prediction_failed",
                phase="phase2", error=str(exc),
            )

    _apply_model_b_filter(prediction_result, predictor, loader, target_date, log_ctx)

    has_bets = bool(prediction_result.strategy.recommended_bets)
    total_bet = prediction_result.strategy.total_recommended_bet

    slack_sent = False
    if slack_webhook_url and not dry_run:
        slack_sent = send_prediction_to_slack(
            slack_webhook_url, prediction_result,
            mention=slack_mention if has_bets else "",
        )
    elif dry_run:
        _log_dry_run(prediction_result)

    if save:
        try:
            save_prediction(prediction_result, decision_time, session_factory, repo)
        except Exception as exc:  # noqa: BLE001
            log_ctx.error("phase2_save_failed", error=str(exc))

    status = "notified" if slack_sent else ("dry_run" if dry_run else "no_webhook")
    log_ctx.info(
        "phase2_race_processed", status=status, has_bets=has_bets,
        total_bet=total_bet, confidence=prediction_result.strategy.confidence_score,
        skip_reason=prediction_result.strategy.skip_reason,
    )

    return TickResult(
        race_id=candidate.race_id, track_name=track_name,
        race_number=race_number, status=status, phase="phase2",
        has_recommended_bets=has_bets, total_recommended_bet=total_bet, slack_sent=slack_sent,
    )


def _rebuild_prediction_from_cache(
    bundle: RacePredictionBundle,
    candidate: RaceDeadlineCandidate,
    session_factory,
    repo: Repository,
    config: StrategyConfig,
    decision_time: datetime,
) -> PredictionResult:
    """Phase 1 でキャッシュした bundle + Phase 2 の最新オッズで戦略を再計算する。

    予測出力（スコア・勝率）は Phase 1 と同一。オッズのみ最新に更新される。
    """
    with session_factory() as session:
        odds_rows = repo.get_latest_market_odds(session, bundle.race_id, judgment_time=decision_time)

    market_odds = market_odds_from_rows(odds_rows)
    strategy = run_strategy(
        bundle, market_odds,
        decision_context=DecisionContext(
            judgment_time=decision_time,
            evaluation_mode=EvaluationMode.LIVE,
            timezone="UTC",
            provenance="autobet.phase2",
        ),
        config=config,
    )

    track_code = TrackCode(candidate.track_id)
    return PredictionResult(
        race_id=bundle.race_id,
        track_code=track_code,
        race_number=candidate.race_number,
        target_date=_extract_date(candidate),
        bundle=bundle,
        strategy=strategy,
        program_sync_warning=None,
        missing_trial_positions=[],
        has_market_odds=bool(market_odds),
    )


# ---------------------------------------------------------------------------
# Morning sync
# ---------------------------------------------------------------------------

def run_morning_sync(
    *,
    session_factory,
    repo: Repository,
    loader: DataLoader,
    target_date: date | None = None,
    tracks: list[TrackCode] | None = None,
    settings: Settings | None = None,
) -> None:
    """当日の全開催をスクレイプ + スケジュール同期 + 履歴キャッシュ構築。"""
    settings = settings or get_settings()
    if target_date is None:
        target_date = datetime.now(tz=JST).date()

    target_tracks = tracks or list(TrackCode)
    log.info("morning_sync_start", date=str(target_date), tracks=[t.name for t in target_tracks])
    start = time.monotonic()

    for track in target_tracks:
        try:
            _scrape_day_for_track(track, target_date, session_factory, repo, settings)
        except Exception as exc:  # noqa: BLE001
            log.warning("morning_sync_scrape_failed", track=track.name, error=str(exc))

    _sync_schedule(target_date, target_tracks, session_factory, repo, settings)
    _build_history_cache(loader, target_date)

    elapsed = time.monotonic() - start
    log.info("morning_sync_done", elapsed_sec=round(elapsed, 1))


# ---------------------------------------------------------------------------
# Daily overview (batch Phase 1)
# ---------------------------------------------------------------------------

def run_daily_overview(
    *,
    session_factory,
    repo: Repository,
    predictor: Predictor,
    loader: DataLoader,
    config: StrategyConfig,
    settings: Settings | None = None,
    slack_webhook_url: str | None = None,
    target_date: date | None = None,
    dry_run: bool = False,
) -> None:
    """全レースを一括予測し、信頼度サマリーを Slack に送信する。

    predict_races() で全レースの特徴量構築を 1 パスにまとめるため、
    レースごとに predict するより大幅に高速（20レース: ~50分 → ~3-5分）。
    """
    from sqlalchemy import select

    from providence.database.tables import Race
    from providence.notification.slack import RaceOverviewEntry, send_daily_overview_to_slack
    from providence.strategy.confidence import race_confidence

    settings = settings or get_settings()
    if target_date is None:
        target_date = datetime.now(tz=JST).date()

    with session_factory() as session:
        races = list(session.execute(
            select(Race).where(Race.race_date == target_date).order_by(Race.track_id, Race.race_number)
        ).scalars())

    if not races:
        log.info("daily_overview_no_races", date=str(target_date))
        return

    race_lookup = {r.id: r for r in races}
    log.info("daily_overview_start", date=str(target_date), races=len(races))

    _load_history_with_cache(predictor, loader, target_date)

    t0 = time.monotonic()
    all_race_df = loader.load_race_dataset(start_date=target_date, end_date=target_date)
    if all_race_df.is_empty():
        log.warning("daily_overview_no_data")
        return

    bundles = predictor.predict_races(all_race_df)
    elapsed = time.monotonic() - t0
    log.info("daily_overview_predicted", races=len(bundles), elapsed_sec=round(elapsed, 1))

    entries: list[RaceOverviewEntry] = []
    for race_id, bundle in bundles.items():
        race = race_lookup.get(race_id)
        if race is None:
            continue
        track_code = TrackCode(race.track_id)
        confidence = race_confidence(bundle)

        top3 = []
        for idx, prob in sorted(bundle.ticket_probs["win"].items(), key=lambda x: x[1], reverse=True)[:3]:
            car = bundle.index_map.post_position_for_index(idx)
            top3.append((car, prob))

        entries.append(RaceOverviewEntry(
            track_name=track_code.japanese_name,
            race_number=race.race_number,
            scheduled_start=race.scheduled_start_at,
            confidence=confidence,
            is_candidate=confidence >= config.min_confidence,
            top3_win_probs=top3,
        ))

    non_candidate_race_ids = set()
    for race_id, bundle in bundles.items():
        race = race_lookup.get(race_id)
        if race is None:
            continue
        confidence = race_confidence(bundle)
        if confidence < config.min_confidence:
            non_candidate_race_ids.add(race_id)
    register_overview_non_candidates(non_candidate_race_ids)

    n_candidates = sum(1 for e in entries if e.is_candidate)
    log.info("daily_overview_done", total=len(entries), candidates=n_candidates)

    if slack_webhook_url and not dry_run:
        send_daily_overview_to_slack(slack_webhook_url, str(target_date), entries, config.min_confidence)
    else:
        for e in entries:
            mark = "✅" if e.is_candidate else "❌"
            log.info(
                "daily_overview_entry", mark=mark, track=e.track_name,
                race=e.race_number, conf=round(e.confidence, 2),
            )


# ---------------------------------------------------------------------------
# Bundle cache (Phase 1 → Phase 2 handoff)
# ---------------------------------------------------------------------------

_phase1_status: dict[int, bool] = {}
_overview_non_candidates: set[int] = set()


def register_overview_non_candidates(race_ids: set[int]) -> None:
    """朝のサマリーで非推奨だったレース ID を登録する。"""
    _overview_non_candidates.update(race_ids)


def _was_previously_non_candidate(race_id: int) -> bool:
    return race_id in _overview_non_candidates


def _mark_phase1_done(race_id: int, *, notify_sent: bool) -> None:
    _phase1_status[race_id] = notify_sent


def _phase1_already_done(race_id: int) -> bool:
    return race_id in _phase1_status


def _save_bundle_cache(race_id: int, bundle: RacePredictionBundle) -> None:
    BUNDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = BUNDLE_CACHE_DIR / f"bundle_{race_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def _load_bundle_cache(race_id: int) -> RacePredictionBundle | None:
    path = BUNDLE_CACHE_DIR / f"bundle_{race_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


# ---------------------------------------------------------------------------
# History cache
# ---------------------------------------------------------------------------

def _history_cache_path(target_date: date) -> Path:
    return HISTORY_CACHE_DIR / f"history_{target_date.isoformat()}.parquet"


def _build_history_cache(loader: DataLoader, target_date: date) -> None:
    cache_path = _history_cache_path(target_date)
    if cache_path.exists():
        return
    cutoff = FeaturePipeline.history_end_for_date(target_date)
    log.info("history_cache_building", cutoff=str(cutoff))
    t0 = time.monotonic()
    history = loader.load_race_dataset(end_date=cutoff)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    history.write_parquet(cache_path)
    log.info("history_cache_built", rows=history.height, elapsed_sec=round(time.monotonic() - t0, 1))


def _load_history_with_cache(predictor: Predictor, loader: DataLoader, target_date: date) -> None:
    cache_path = _history_cache_path(target_date)
    predictor.load_history_from_parquet(str(cache_path), target_date)


# ---------------------------------------------------------------------------
# Phase tracking (DB-based for Phase 2)
# ---------------------------------------------------------------------------

def _phase2_already_done(session: Session, race_id: int) -> bool:
    from sqlalchemy import select

    from providence.database.tables import StrategyRun

    existing = session.execute(
        select(StrategyRun).where(
            StrategyRun.race_id == race_id,
            StrategyRun.evaluation_mode == "live",
        )
    ).scalar_one_or_none()
    return existing is not None


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _scrape_entries_and_conditions(
    track_code: TrackCode, race_date: date, race_number: int,
    session_factory, repo: Repository, settings: Settings,
) -> None:
    async def _inner():
        async with AutoraceJpScraper(settings) as scraper:
            entries = await scraper.get_race_entries(track_code, race_date, race_number)
            conditions = await scraper.get_live_conditions(track_code)
        return entries, conditions

    entries, conditions = asyncio.run(_inner())
    with session_factory() as session:
        if entries.entries:
            repo.save_race_data(session, entries, None, update_race_metadata=False)
        if conditions and any(v is not None for v in conditions.values()):
            repo.update_race_conditions(
                session, track_code.value, race_date, conditions, force=True, race_number=race_number,
            )


def _scrape_odds_for_race(
    track_code: TrackCode, race_date: date, race_number: int,
    session_factory, repo: Repository, settings: Settings,
) -> None:
    async def _inner():
        async with AutoraceJpScraper(settings) as scraper:
            entries = await scraper.get_race_entries(track_code, race_date, race_number)
            odds = await scraper.get_odds(track_code, race_date, race_number)
            conditions = await scraper.get_live_conditions(track_code)
        return entries, odds, conditions

    entries, odds, conditions = asyncio.run(_inner())
    with session_factory() as session:
        if entries.entries:
            repo.save_race_data(session, entries, None, update_race_metadata=False)
        if conditions and any(v is not None for v in conditions.values()):
            repo.update_race_conditions(
                session, track_code.value, race_date, conditions, force=True, race_number=race_number,
            )
        if odds.odds:
            race = repo.get_race(session, track_code.value, race_date, race_number)
            if race is not None:
                repo.save_odds(
                    session, race.id, odds.odds,
                    captured_at=datetime.combine(race_date, datetime.min.time()),
                    source_name="autorace_jp",
                )


def _scrape_day_for_track(
    track_code: TrackCode, target_date: date, session_factory, repo: Repository, settings: Settings,
) -> None:
    async def _inner():
        async with AutoraceJpScraper(settings) as scraper:
            races = []
            for race_no in range(1, 13):
                try:
                    entries = await scraper.get_race_entries(track_code, target_date, race_no)
                    if entries.entries:
                        races.append(entries)
                except Exception:  # noqa: BLE001
                    break
            return races

    race_entries_list = asyncio.run(_inner())
    with session_factory() as session:
        repo.ensure_tracks(session)
        for entries in race_entries_list:
            repo.save_race_data(session, entries, None)
    log.info("scrape_day_done", track=track_code.name, races=len(race_entries_list))


def _sync_schedule(
    target_date: date, tracks: list[TrackCode], session_factory, repo: Repository, settings: Settings,
) -> None:
    async def _inner():
        async with OddsparkScraper(settings) as scraper:
            with session_factory() as session:
                return await sync_race_schedule(session, scraper, target_date, tracks=tracks)

    result = asyncio.run(_inner())
    log.info("schedule_sync_done", updated=result.updated, skipped=result.skipped, errors=result.errors)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_date(candidate: RaceDeadlineCandidate) -> date:
    return candidate.telvote_close_at.date() if candidate.telvote_close_at else datetime.now(tz=JST).date()


def _build_forced_candidates(
    session: Session, repo: Repository, track_code: TrackCode, race_number: int,
) -> list[RaceDeadlineCandidate]:
    from sqlalchemy import select

    from providence.database.tables import Race

    now = datetime.now(tz=JST).replace(tzinfo=None)
    target_date = datetime.now(tz=JST).date()
    race = session.execute(
        select(Race).where(
            Race.track_id == track_code.value, Race.race_date == target_date, Race.race_number == race_number,
        )
    ).scalar_one_or_none()
    if race is None:
        log.warning("forced_race_not_found", track=track_code.name, race=race_number)
        return []
    return [RaceDeadlineCandidate(
        race_id=race.id, track_id=track_code.value, race_number=race_number,
        telvote_close_at=race.telvote_close_at or now, scheduled_start_at=race.scheduled_start_at or now,
        minutes_until_close=0.0,
    )]


def _apply_model_b_filter(
    result: PredictionResult, predictor: Predictor, loader: DataLoader, target_date: date, log_ctx,
) -> None:
    from providence.model.model_b_filter import ModelBFilter
    from providence.model.store import ModelStore

    store = ModelStore()
    version_dir = store.version_dir(predictor.model_version)
    model_b_filter = ModelBFilter.load(version_dir)
    if model_b_filter is None or model_b_filter.threshold is None:
        return
    try:
        import polars as pl

        race_df = loader.load_race_dataset(start_date=target_date, end_date=target_date).filter(
            pl.col("race_id") == result.race_id
        )
        if race_df.is_empty():
            return
        if predictor._is_ensemble and predictor._models is not None:
            model_a_booster = predictor._models.get("lambdarank") or next(iter(predictor._models.values()))
        elif predictor.model is not None:
            model_a_booster = predictor.model
        else:
            return
        passes, quality = model_b_filter.should_bet(race_df, model_a_booster, predictor.metadata["feature_columns"])
        if not passes:
            log_ctx.info("model_b_filtered", race_quality=round(quality, 4))
            result.strategy.recommended_bets = []
            if not result.strategy.skip_reason:
                result.strategy.skip_reason = "model_b_filtered"
    except Exception as exc:  # noqa: BLE001
        log_ctx.warning("model_b_filter_error", error=str(exc))


def _log_dry_run(result: PredictionResult) -> None:
    from providence.notification.slack import build_slack_message

    msg = build_slack_message(result)
    log.info("autobet_dry_run", message_text=msg.get("text", ""))
