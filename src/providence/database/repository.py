"""Data access layer with transactional writes and business validation."""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

import structlog
from sqlalchemy import desc, func, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from providence.database.tables import (
    BettingLog,
    FeedbackRun,
    ModelPerformance,
    OddsSnapshot,
    Prediction,
    Race,
    RaceEntry,
    RaceResult,
    Rider,
    ScrapeLog,
    StrategyRun,
    TicketPayout,
    Track,
)
from providence.domain.enums import TrackCode
from providence.scraper.schemas import (
    EntryRow,
    OddsRow,
    PlayerProfileResponse,
    PlayerSummary,
    RaceEntriesResponse,
    RaceResultResponse,
    RefundRow,
    ResultRow,
)

logger = structlog.get_logger()

_TRACK_DATA: dict[int, tuple[str, str]] = {
    TrackCode.KAWAGUCHI: ("川口", "埼玉県川口市"),
    TrackCode.ISESAKI: ("伊勢崎", "群馬県伊勢崎市"),
    TrackCode.HAMAMATSU: ("浜松", "静岡県浜松市"),
    TrackCode.IIZUKA: ("飯塚", "福岡県飯塚市"),
    TrackCode.SANYO: ("山陽", "山口県山陽小野田市"),
}


class Repository:
    """Transactional data access for race data."""

    def __init__(self) -> None:
        self._log = logger.bind(component="repository")

    # ------------------------------------------------------------------ #
    #  High-level transactional methods
    # ------------------------------------------------------------------ #

    def save_race_data(
        self,
        session: Session,
        entries_resp: RaceEntriesResponse,
        result_resp: RaceResultResponse | None = None,
    ) -> Race:
        """Save one race atomically (entries + optional results + refunds)."""
        with session.begin():
            race = self._upsert_race(session, entries_resp)
            self._upsert_riders_from_entries(session, entries_resp.entries)
            self._upsert_race_entries(session, race, entries_resp.entries)

            if result_resp:
                self._update_race_conditions(session, race, result_resp)
                self._insert_race_results(session, race, result_resp.results)
                self._insert_ticket_payouts(session, race, result_resp.refunds)

        return race

    def save_odds(
        self,
        session: Session,
        race_id: int,
        odds: list[OddsRow],
        *,
        captured_at: datetime | None = None,
        ingestion_batch_id: str | None = None,
        source_name: str | None = None,
    ) -> int:
        """Save market odds data for a race. Returns count of rows inserted."""
        batch_id = ingestion_batch_id or str(uuid4())
        captured_time = captured_at or datetime.now()
        transaction = session.begin_nested() if session.in_transaction() else session.begin()
        with transaction:
            count = 0
            for o in odds:
                snapshot = OddsSnapshot(
                    race_id=race_id,
                    ticket_type=o.ticket_type.value,
                    combination=o.combination,
                    odds_value=o.odds_value,
                    popularity=o.popularity,
                    ingestion_batch_id=batch_id,
                    source_name=source_name,
                    captured_at=captured_time,
                )
                session.add(snapshot)
                count += 1
            return count

    def get_race(self, session: Session, track_id: int, race_date: date, race_number: int) -> Race | None:
        """Fetch one race by identity."""
        return session.execute(
            select(Race).where(
                Race.track_id == track_id,
                Race.race_date == race_date,
                Race.race_number == race_number,
            )
        ).scalar_one_or_none()

    def get_latest_market_odds(
        self,
        session: Session,
        race_id: int,
        *,
        judgment_time: datetime,
    ) -> list[OddsSnapshot]:
        """Return the latest complete market-odds batch before judgment time.

        Legacy rows without ``ingestion_batch_id`` are ignored because they may
        contain mixed payout data from the pre-migration schema.
        """
        latest_row = session.execute(
            select(OddsSnapshot)
            .where(
                OddsSnapshot.race_id == race_id,
                OddsSnapshot.ingestion_batch_id.is_not(None),
                OddsSnapshot.captured_at <= judgment_time,
            )
            .order_by(desc(OddsSnapshot.captured_at), desc(OddsSnapshot.id))
            .limit(1)
        ).scalar_one_or_none()
        if latest_row is None or latest_row.ingestion_batch_id is None:
            return []

        return list(
            session.execute(
                select(OddsSnapshot)
                .where(
                    OddsSnapshot.race_id == race_id,
                    OddsSnapshot.ingestion_batch_id == latest_row.ingestion_batch_id,
                )
                .order_by(OddsSnapshot.ticket_type, OddsSnapshot.combination)
            ).scalars()
        )

    def get_latest_market_odds_for_races(
        self,
        session: Session,
        race_ids: list[int],
        *,
        judgment_time: datetime,
    ) -> dict[int, list[OddsSnapshot]]:
        """Return latest market-odds batches for multiple races."""
        if not race_ids:
            return {}
        eligible_rows = list(
            session.execute(
                select(OddsSnapshot)
                .where(
                    OddsSnapshot.race_id.in_(race_ids),
                    OddsSnapshot.ingestion_batch_id.is_not(None),
                    OddsSnapshot.captured_at <= judgment_time,
                )
                .order_by(OddsSnapshot.race_id, desc(OddsSnapshot.captured_at), desc(OddsSnapshot.id))
            ).scalars()
        )

        selected_batch_by_race: dict[int, str] = {}
        for row in eligible_rows:
            race_id = int(row.race_id)
            if race_id not in selected_batch_by_race and row.ingestion_batch_id is not None:
                selected_batch_by_race[race_id] = row.ingestion_batch_id

        grouped: dict[int, list[OddsSnapshot]] = {race_id: [] for race_id in race_ids}
        for row in eligible_rows:
            race_id = int(row.race_id)
            if selected_batch_by_race.get(race_id) == row.ingestion_batch_id:
                grouped.setdefault(race_id, []).append(row)

        for race_id, rows in grouped.items():
            rows.sort(key=lambda item: (item.ticket_type, item.combination))
        return grouped

    def get_ticket_payouts(self, session: Session, race_id: int) -> list[TicketPayout]:
        """Return settled payouts for a race."""
        return list(
            session.execute(
                select(TicketPayout)
                .where(TicketPayout.race_id == race_id)
                .order_by(TicketPayout.ticket_type, TicketPayout.combination)
            ).scalars()
        )

    def get_ticket_payouts_for_races(self, session: Session, race_ids: list[int]) -> dict[int, list[TicketPayout]]:
        """Return settled payouts for multiple races."""
        if not race_ids:
            return {}
        rows = list(
            session.execute(
                select(TicketPayout)
                .where(TicketPayout.race_id.in_(race_ids))
                .order_by(TicketPayout.race_id, TicketPayout.ticket_type, TicketPayout.combination)
            ).scalars()
        )
        grouped: dict[int, list[TicketPayout]] = {race_id: [] for race_id in race_ids}
        for row in rows:
            grouped.setdefault(int(row.race_id), []).append(row)
        return grouped

    def save_strategy_run(
        self,
        session: Session,
        strategy_run: StrategyRun,
        predictions: list[Prediction],
    ) -> StrategyRun:
        """Persist one strategy run and its prediction legs."""
        with session.begin():
            session.add(strategy_run)
            session.flush()
            for prediction in predictions:
                prediction.strategy_run_id = strategy_run.id
                session.add(prediction)
            session.flush()
            return strategy_run

    def get_latest_strategy_runs(
        self,
        session: Session,
        *,
        race_ids: list[int] | None = None,
        since_date: date | None = None,
    ) -> list[StrategyRun]:
        """Return the latest StrategyRun per race by judgment and creation time."""
        query = select(StrategyRun)
        if race_ids:
            query = query.where(StrategyRun.race_id.in_(race_ids))
        if since_date is not None:
            query = query.where(StrategyRun.judgment_time >= datetime.combine(since_date, datetime.min.time()))

        runs = list(
            session.execute(
                query.order_by(
                    StrategyRun.race_id,
                    desc(StrategyRun.judgment_time),
                    desc(StrategyRun.created_at),
                    desc(StrategyRun.id),
                )
            ).scalars()
        )
        latest_by_race: dict[int, StrategyRun] = {}
        for run in runs:
            latest_by_race.setdefault(int(run.race_id), run)
        return list(latest_by_race.values())

    def get_predictions_for_strategy_runs(self, session: Session, strategy_run_ids: list[int]) -> list[Prediction]:
        if not strategy_run_ids:
            return []
        return list(
            session.execute(
                select(Prediction)
                .where(Prediction.strategy_run_id.in_(strategy_run_ids))
                .order_by(Prediction.strategy_run_id, Prediction.id)
            ).scalars()
        )

    def get_betting_logs_for_prediction_ids(self, session: Session, prediction_ids: list[int]) -> list[BettingLog]:
        if not prediction_ids:
            return []
        return list(
            session.execute(
                select(BettingLog)
                .where(BettingLog.prediction_id.in_(prediction_ids))
                .order_by(BettingLog.prediction_id)
            ).scalars()
        )

    def upsert_betting_logs(self, session: Session, betting_logs: list[dict]) -> int:
        """Upsert paper-trade betting logs by prediction_id."""
        if not betting_logs:
            return 0

        transaction = session.begin_nested() if session.in_transaction() else session.begin()
        with transaction:
            for row in betting_logs:
                stmt = sqlite_insert(BettingLog).values(**row)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["prediction_id"],
                    set_={
                        "race_id": row["race_id"],
                        "actual_bet_amount": row["actual_bet_amount"],
                        "payout": row["payout"],
                        "profit": row["profit"],
                        "bankroll_after": row.get("bankroll_after"),
                        "reconciled_at": row["reconciled_at"],
                    },
                )
                session.execute(stmt)
        return len(betting_logs)

    def upsert_model_performance(self, session: Session, rows: list[dict]) -> int:
        """Upsert model performance rows by model/version/window identity."""
        if not rows:
            return 0
        transaction = session.begin_nested() if session.in_transaction() else session.begin()
        with transaction:
            for row in rows:
                stmt = sqlite_insert(ModelPerformance).values(**row)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["model_version", "evaluation_date", "window"],
                    set_={
                        "sample_size": row["sample_size"],
                        "win_accuracy": row.get("win_accuracy"),
                        "top3_accuracy": row.get("top3_accuracy"),
                        "brier_score": row.get("brier_score"),
                        "ndcg": row.get("ndcg"),
                        "roi": row.get("roi"),
                        "calibration_error": row.get("calibration_error"),
                        "computed_at": row["computed_at"],
                    },
                )
                session.execute(stmt)
        return len(rows)

    def log_feedback_run(
        self,
        session: Session,
        *,
        job_name: str,
        model_version: str | None = None,
        evaluation_date: date | None = None,
        status: str = "success",
        details: str | None = None,
        executed_at: datetime | None = None,
    ) -> None:
        transaction = session.begin_nested() if session.in_transaction() else session.begin()
        with transaction:
            session.add(
                FeedbackRun(
                    job_name=job_name,
                    model_version=model_version,
                    evaluation_date=evaluation_date,
                    status=status,
                    details=details,
                    executed_at=executed_at or datetime.now(),
                )
            )

    def save_player(self, session: Session, profile: PlayerProfileResponse | PlayerSummary) -> Rider:
        """Upsert a single rider from profile data."""
        with session.begin():
            return self._upsert_rider_from_profile(session, profile)

    def update_race_conditions(
        self,
        session: Session,
        track_id: int,
        race_date: date,
        conditions: dict[str, object],
    ) -> int:
        """Update weather/track conditions for all races on a given day at a track.

        Returns count of races updated.
        """
        with session.begin():
            races = session.execute(
                select(Race).where(Race.track_id == track_id, Race.race_date == race_date)
            ).scalars().all()

            count = 0
            for race in races:
                updated = False
                if conditions.get("weather") and not race.weather:
                    race.weather = conditions["weather"]
                    updated = True
                if conditions.get("track_condition") and not race.track_condition:
                    tc = conditions["track_condition"]
                    race.track_condition = tc.value if hasattr(tc, "value") else str(tc)
                    updated = True
                if conditions.get("temperature") is not None and race.temperature is None:
                    race.temperature = float(conditions["temperature"])
                    updated = True
                if conditions.get("humidity") is not None and race.humidity is None:
                    race.humidity = float(conditions["humidity"])
                    updated = True
                if conditions.get("track_temperature") is not None and race.track_temperature is None:
                    race.track_temperature = float(conditions["track_temperature"])
                    updated = True
                if updated:
                    count += 1

            return count

    # ------------------------------------------------------------------ #
    #  Seed data
    # ------------------------------------------------------------------ #

    def ensure_tracks(self, session: Session) -> None:
        """Insert track master data if not present."""
        with session.begin():
            for track_id, (name, location) in _TRACK_DATA.items():
                stmt = sqlite_insert(Track).values(id=track_id, name=name, location=location)
                stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
                session.execute(stmt)

    # ------------------------------------------------------------------ #
    #  Resume / query helpers
    # ------------------------------------------------------------------ #

    def get_collected_races(
        self, session: Session, start_date: date, end_date: date, track_id: int | None = None
    ) -> set[tuple[int, date, int]]:
        """Return set of (track_id, race_date, race_number) already in DB."""
        query = (
            select(Race.track_id, Race.race_date, Race.race_number)
            .where(Race.race_date.between(start_date, end_date))
            .where(Race.entries.any())
        )
        if track_id is not None:
            query = query.where(Race.track_id == track_id)

        rows = session.execute(query).all()
        return {(r[0], r[1], r[2]) for r in rows}

    def get_db_stats(self, session: Session) -> dict[str, int]:
        """Return basic DB statistics."""
        stats = {}
        for table_name in (
            "tracks",
            "riders",
            "races",
            "race_entries",
            "race_results",
            "odds_snapshot",
            "ticket_payouts",
            "predictions",
            "strategy_runs",
            "scrape_log",
        ):
            result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))  # noqa: S608
            stats[table_name] = result.scalar() or 0
        return stats

    def get_recent_scrape_logs(self, session: Session, limit: int = 20) -> list[ScrapeLog]:
        """Return recent scrape logs ordered by newest first."""
        return list(
            session.execute(
                select(ScrapeLog).order_by(ScrapeLog.executed_at.desc()).limit(limit)
            ).scalars()
        )

    def get_recent_model_performance(
        self,
        session: Session,
        *,
        model_version: str | None = None,
        limit: int = 20,
    ) -> list[ModelPerformance]:
        query = select(ModelPerformance)
        if model_version is not None:
            query = query.where(ModelPerformance.model_version == model_version)
        return list(
            session.execute(
                query.order_by(
                    ModelPerformance.evaluation_date.desc(),
                    ModelPerformance.window,
                    ModelPerformance.computed_at.desc(),
                ).limit(limit)
            ).scalars()
        )

    def get_feedback_freshness(self, session: Session) -> dict[str, datetime | None]:
        latest_reconcile = session.execute(
            select(func.max(FeedbackRun.executed_at)).where(FeedbackRun.job_name == "reconcile")
        ).scalar()
        latest_performance = session.execute(
            select(func.max(FeedbackRun.executed_at)).where(FeedbackRun.job_name == "performance")
        ).scalar()
        latest_drift = session.execute(
            select(func.max(FeedbackRun.executed_at)).where(FeedbackRun.job_name == "drift")
        ).scalar()
        return {
            "latest_reconcile_at": latest_reconcile,
            "latest_performance_at": latest_performance,
            "latest_drift_at": latest_drift,
        }

    def get_strategy_run_summary(self, session: Session) -> dict[str, object]:
        """Return high-level operational summary for strategy outputs."""
        total_runs = session.execute(select(func.count()).select_from(StrategyRun)).scalar() or 0
        total_predictions = session.execute(select(func.count()).select_from(Prediction)).scalar() or 0
        latest_run = session.execute(
            select(StrategyRun).order_by(StrategyRun.created_at.desc(), StrategyRun.id.desc()).limit(1)
        ).scalar_one_or_none()
        latest_prediction_at = session.execute(select(func.max(Prediction.predicted_at))).scalar()

        return {
            "strategy_runs": int(total_runs),
            "predictions": int(total_predictions),
            "latest_run": latest_run,
            "latest_prediction_at": latest_prediction_at,
        }

    def backfill_legacy_payouts(self, session: Session) -> int:
        """Copy legacy payout rows from odds_snapshot into ticket_payouts.

        Rows without ``ingestion_batch_id`` are treated as pre-Phase-3 legacy
        payout data. The operation is idempotent.
        """
        transaction = session.begin_nested() if session.in_transaction() else session.begin()
        with transaction:
            before = session.execute(text("SELECT COUNT(*) FROM ticket_payouts")).scalar() or 0
            session.execute(
                text(
                    """
                    INSERT INTO ticket_payouts (
                        race_id,
                        ticket_type,
                        combination,
                        payout_value,
                        popularity,
                        settled_at
                    )
                    SELECT
                        o.race_id,
                        o.ticket_type,
                        o.combination,
                        o.odds_value,
                        o.popularity,
                        o.captured_at
                    FROM odds_snapshot o
                    WHERE o.ingestion_batch_id IS NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM ticket_payouts p
                          WHERE p.race_id = o.race_id
                            AND p.ticket_type = o.ticket_type
                            AND p.combination = o.combination
                      )
                    """
                )
            )
            after = session.execute(text("SELECT COUNT(*) FROM ticket_payouts")).scalar() or 0
        return int(after - before)

    # ------------------------------------------------------------------ #
    #  Scrape log
    # ------------------------------------------------------------------ #

    def log_scrape(
        self,
        session: Session,
        *,
        source: str,
        target: str,
        target_date: date | None = None,
        track_id: int | None = None,
        records_count: int = 0,
        status: str = "success",
        error_message: str | None = None,
        duration_sec: float | None = None,
    ) -> None:
        with session.begin():
            log_entry = ScrapeLog(
                source=source,
                target=target,
                target_date=target_date,
                track_id=track_id,
                records_count=records_count,
                status=status,
                error_message=error_message,
                duration_sec=duration_sec,
            )
            session.add(log_entry)

    # ------------------------------------------------------------------ #
    #  Internal: upsert / insert helpers
    # ------------------------------------------------------------------ #

    def _upsert_race(self, session: Session, resp: RaceEntriesResponse) -> Race:
        existing = session.execute(
            select(Race).where(
                Race.track_id == resp.track.value,
                Race.race_date == resp.race_date,
                Race.race_number == resp.race_number,
            )
        ).scalar_one_or_none()

        if existing:
            existing.grade = resp.grade.value
            existing.title = resp.title
            existing.distance = resp.distance
            existing.weather = resp.weather
            existing.track_condition = resp.track_condition.value if resp.track_condition else None
            existing.temperature = resp.temperature
            existing.humidity = resp.humidity
            existing.track_temperature = resp.track_temperature
            return existing

        race = Race(
            track_id=resp.track.value,
            race_date=resp.race_date,
            race_number=resp.race_number,
            grade=resp.grade.value,
            title=resp.title,
            distance=resp.distance,
            track_condition=resp.track_condition.value if resp.track_condition else None,
            weather=resp.weather,
            temperature=resp.temperature,
            humidity=resp.humidity,
            track_temperature=resp.track_temperature,
        )
        session.add(race)
        session.flush()
        return race

    def _update_race_conditions(self, session: Session, race: Race, resp: RaceResultResponse) -> None:
        """Update race weather/track info from result (may have more complete data)."""
        if resp.weather:
            race.weather = resp.weather
        if resp.track_condition:
            race.track_condition = resp.track_condition.value
        if resp.temperature is not None:
            race.temperature = resp.temperature
        if resp.humidity is not None:
            race.humidity = resp.humidity
        if resp.track_temperature is not None:
            race.track_temperature = resp.track_temperature

    def _upsert_riders_from_entries(self, session: Session, entries: list[EntryRow]) -> None:
        for entry in entries:
            self._upsert_rider(
                session,
                registration_number=entry.rider_registration_number,
                name=entry.rider_name,
                generation=entry.generation,
                rank=entry.rank,
            )

    def _upsert_rider(
        self,
        session: Session,
        *,
        registration_number: str,
        name: str,
        generation: int | None = None,
        rank: object | None = None,
        **kwargs: object,
    ) -> Rider:
        existing = session.execute(
            select(Rider).where(Rider.registration_number == registration_number)
        ).scalar_one_or_none()

        if existing:
            if name:
                existing.name = name
            if generation is not None:
                existing.generation = generation
            return existing

        rider = Rider(registration_number=registration_number, name=name, generation=generation)
        session.add(rider)
        session.flush()
        return rider

    def _upsert_rider_from_profile(self, session: Session, profile: PlayerProfileResponse | PlayerSummary) -> Rider:
        existing = session.execute(
            select(Rider).where(Rider.registration_number == profile.registration_number)
        ).scalar_one_or_none()

        if existing:
            existing.name = profile.name
            if hasattr(profile, "name_kana") and profile.name_kana:
                existing.name_kana = profile.name_kana
            if hasattr(profile, "birth_year") and profile.birth_year:
                existing.birth_year = profile.birth_year
            if profile.generation is not None:
                existing.generation = profile.generation
            if profile.home_track is not None:
                existing.home_track_id = profile.home_track.value
            return existing

        rider = Rider(
            registration_number=profile.registration_number,
            name=profile.name,
            name_kana=getattr(profile, "name_kana", None),
            birth_year=getattr(profile, "birth_year", None),
            generation=profile.generation,
            home_track_id=profile.home_track.value if profile.home_track else None,
        )
        session.add(rider)
        session.flush()
        return rider

    def _upsert_race_entries(self, session: Session, race: Race, entries: list[EntryRow]) -> None:
        self._validate_entries(entries)

        for entry in entries:
            rider = session.execute(
                select(Rider).where(Rider.registration_number == entry.rider_registration_number)
            ).scalar_one_or_none()

            if not rider:
                self._log.warning("rider_not_found", registration_number=entry.rider_registration_number)
                continue

            existing = session.execute(
                select(RaceEntry).where(
                    RaceEntry.race_id == race.id,
                    RaceEntry.post_position == entry.post_position,
                )
            ).scalar_one_or_none()

            if existing:
                existing.rider_id = rider.id
                existing.handicap_meters = entry.handicap_meters
                existing.trial_time = entry.trial_time
                existing.avg_trial_time = entry.avg_trial_time
                existing.trial_deviation = entry.trial_deviation
                existing.race_score = entry.race_score
            else:
                race_entry = RaceEntry(
                    race_id=race.id,
                    rider_id=rider.id,
                    post_position=entry.post_position,
                    handicap_meters=entry.handicap_meters,
                    trial_time=entry.trial_time,
                    avg_trial_time=entry.avg_trial_time,
                    trial_deviation=entry.trial_deviation,
                    race_score=entry.race_score,
                )
                session.add(race_entry)

    def _insert_race_results(self, session: Session, race: Race, results: list[ResultRow]) -> None:
        for r in results:
            self._validate_result(r, len(results))

            entry = session.execute(
                select(RaceEntry).where(
                    RaceEntry.race_id == race.id,
                    RaceEntry.post_position == r.post_position,
                )
            ).scalar_one_or_none()

            if not entry:
                self._log.warning("entry_not_found_for_result", race_id=race.id, post_position=r.post_position)
                continue

            if r.entry_status:
                entry.entry_status = r.entry_status.value

            existing_result = session.execute(
                select(RaceResult).where(RaceResult.race_entry_id == entry.id)
            ).scalar_one_or_none()

            if existing_result:
                existing_result.finish_position = r.finish_position
                existing_result.race_time = r.race_time
                existing_result.start_timing = r.start_timing
                existing_result.accident_code = r.accident_code
            else:
                result = RaceResult(
                    race_entry_id=entry.id,
                    finish_position=r.finish_position,
                    race_time=r.race_time,
                    start_timing=r.start_timing,
                    accident_code=r.accident_code,
                )
                session.add(result)

    def _insert_ticket_payouts(self, session: Session, race: Race, refunds: list[RefundRow]) -> None:
        for refund in refunds:
            payout_value = refund.refund_amount / 100.0
            if payout_value <= 0:
                continue

            existing = session.execute(
                select(TicketPayout).where(
                    TicketPayout.race_id == race.id,
                    TicketPayout.ticket_type == refund.ticket_type.value,
                    TicketPayout.combination == refund.combination,
                )
            ).scalar_one_or_none()

            if existing:
                existing.payout_value = payout_value
                existing.popularity = refund.popularity
                existing.settled_at = datetime.now()
                continue

            payout = TicketPayout(
                race_id=race.id,
                ticket_type=refund.ticket_type.value,
                combination=refund.combination,
                payout_value=payout_value,
                popularity=refund.popularity,
                settled_at=datetime.now(),
            )
            session.add(payout)

    # ------------------------------------------------------------------ #
    #  Validation (warn but don't reject)
    # ------------------------------------------------------------------ #

    def _validate_entries(self, entries: list[EntryRow]) -> None:
        count = len(entries)
        if count < 3 or count > 8:
            self._log.warning("unusual_entry_count", count=count)

        for e in entries:
            if e.trial_time is not None and (e.trial_time < 2.5 or e.trial_time > 5.0):
                self._log.warning("unusual_trial_time", time=e.trial_time, rider=e.rider_registration_number)
            if e.handicap_meters % 10 != 0:
                self._log.warning("unusual_handicap", meters=e.handicap_meters, rider=e.rider_registration_number)

    def _validate_result(self, result: ResultRow, total_entries: int) -> None:
        pos = result.finish_position
        if pos is not None and (pos < 1 or pos > total_entries):
            self._log.warning(
                "unusual_finish_position",
                position=result.finish_position,
                total=total_entries,
                rider=result.rider_registration_number,
            )
