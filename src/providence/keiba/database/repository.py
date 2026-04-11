"""Repository for keiba (JRA horse racing) data persistence.

Follows the same patterns as the autorace Repository:
- Session is passed as method argument
- Upsert: select existing → update or add new
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from providence.keiba.database.tables import (
    KeibaHorse,
    KeibaImportLog,
    KeibaJockey,
    KeibaRace,
    KeibaRacecourse,
    KeibaRaceEntry,
    KeibaRaceResult,
    KeibaTicketPayout,
    KeibaTrainer,
)

logger = structlog.get_logger()

_JRA_RACECOURSES = [
    (1, "札幌", "Sapporo"),
    (2, "函館", "Hakodate"),
    (3, "福島", "Fukushima"),
    (4, "新潟", "Niigata"),
    (5, "東京", "Tokyo"),
    (6, "中山", "Nakayama"),
    (7, "中京", "Chukyo"),
    (8, "京都", "Kyoto"),
    (9, "阪神", "Hanshin"),
    (10, "小倉", "Kokura"),
]


class KeibaRepository:
    def __init__(self) -> None:
        self._log = logger.bind(component="KeibaRepository")

    def ensure_racecourses(self, session: Session) -> None:
        for rc_id, name, name_en in _JRA_RACECOURSES:
            existing = session.get(KeibaRacecourse, rc_id)
            if existing is None:
                session.add(KeibaRacecourse(id=rc_id, name=name, name_en=name_en))
        session.flush()

    def save_jockeys(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            code = rec.get("jockey_code")
            if not code:
                continue
            existing = session.execute(
                select(KeibaJockey).where(KeibaJockey.jockey_code == code)
            ).scalar_one_or_none()
            if existing:
                for attr in ("jockey_name", "affiliation_code", "apprentice_class", "retired_flag",
                             "career_flat_1st", "career_flat_2nd", "career_flat_3rd", "career_flat_unplaced"):
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                session.add(KeibaJockey(
                    jockey_code=code,
                    jockey_name=rec.get("jockey_name"),
                    affiliation_code=rec.get("affiliation_code"),
                    apprentice_class=rec.get("apprentice_class"),
                    retired_flag=rec.get("retired_flag"),
                    career_flat_1st=rec.get("career_flat_1st"),
                    career_flat_2nd=rec.get("career_flat_2nd"),
                    career_flat_3rd=rec.get("career_flat_3rd"),
                    career_flat_unplaced=rec.get("career_flat_unplaced"),
                ))
                count += 1
        session.flush()
        return count

    def save_trainers(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            code = rec.get("trainer_code")
            if not code:
                continue
            existing = session.execute(
                select(KeibaTrainer).where(KeibaTrainer.trainer_code == code)
            ).scalar_one_or_none()
            if existing:
                for attr in ("trainer_name", "affiliation_code", "retired_flag",
                             "career_flat_1st", "career_flat_2nd", "career_flat_3rd", "career_flat_unplaced"):
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                session.add(KeibaTrainer(
                    trainer_code=code,
                    trainer_name=rec.get("trainer_name"),
                    affiliation_code=rec.get("affiliation_code"),
                    retired_flag=rec.get("retired_flag"),
                    career_flat_1st=rec.get("career_flat_1st"),
                    career_flat_2nd=rec.get("career_flat_2nd"),
                    career_flat_3rd=rec.get("career_flat_3rd"),
                    career_flat_unplaced=rec.get("career_flat_unplaced"),
                ))
                count += 1
        session.flush()
        return count

    def save_horses(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            brn = rec.get("blood_registration_number")
            if not brn:
                continue
            existing = session.execute(
                select(KeibaHorse).where(KeibaHorse.blood_registration_number == brn)
            ).scalar_one_or_none()
            if existing:
                for attr in ("horse_name", "sex_code", "birth_date", "sire_name", "dam_name",
                             "broodmare_sire_name", "sire_code", "broodmare_sire_code", "retired_flag"):
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                session.add(KeibaHorse(
                    blood_registration_number=brn,
                    horse_name=rec.get("horse_name"),
                    sex_code=rec.get("sex_code"),
                    birth_date=rec.get("birth_date"),
                    sire_name=rec.get("sire_name"),
                    dam_name=rec.get("dam_name"),
                    broodmare_sire_name=rec.get("broodmare_sire_name"),
                    sire_code=rec.get("sire_code"),
                    broodmare_sire_code=rec.get("broodmare_sire_code"),
                    retired_flag=rec.get("retired_flag"),
                ))
                count += 1
        session.flush()
        return count

    def save_races(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            race_key = self._build_race_key(rec)
            if not race_key:
                continue
            existing = session.execute(
                select(KeibaRace).where(KeibaRace.race_key == race_key)
            ).scalar_one_or_none()

            race_date = self._parse_date(rec.get("race_date"))
            racecourse_id = self._parse_racecourse_id(rec.get("place_code"))

            if existing:
                existing.race_date = race_date or existing.race_date
                existing.racecourse_id = racecourse_id or existing.racecourse_id
                for attr in ("distance", "surface_code", "direction_code", "course_type_code",
                             "going_code", "weather_code", "race_name", "class_code",
                             "weight_rule_code", "num_runners", "post_time"):
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                race_number = rec.get("race_number")
                session.add(KeibaRace(
                    race_key=race_key,
                    race_date=race_date,
                    racecourse_id=racecourse_id,
                    race_number=int(race_number) if race_number else None,
                    distance=rec.get("distance"),
                    surface_code=rec.get("surface_code"),
                    direction_code=rec.get("direction_code"),
                    course_type_code=rec.get("course_type_code"),
                    going_code=rec.get("going_code"),
                    weather_code=rec.get("weather_code"),
                    race_name=rec.get("race_name"),
                    class_code=rec.get("class_code"),
                    weight_rule_code=rec.get("weight_rule_code"),
                    num_runners=rec.get("num_runners"),
                    post_time=rec.get("post_time"),
                ))
                count += 1
        session.flush()
        return count

    def save_entries(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            race_key = self._build_race_key(rec)
            if not race_key:
                continue
            race = session.execute(
                select(KeibaRace).where(KeibaRace.race_key == race_key)
            ).scalar_one_or_none()
            if race is None:
                self._log.warning("race_not_found", race_key=race_key)
                continue

            post_pos = rec.get("post_position")
            if post_pos is None:
                continue

            existing = session.execute(
                select(KeibaRaceEntry).where(
                    KeibaRaceEntry.race_id == race.id,
                    KeibaRaceEntry.post_position == post_pos,
                )
            ).scalar_one_or_none()

            horse_id = self._resolve_horse_id(session, rec.get("blood_registration_number"))
            jockey_id = self._resolve_jockey_id(session, rec.get("jockey_code"))
            trainer_id = self._resolve_trainer_id(session, rec.get("trainer_code"))

            _ENTRY_ATTRS = (
                "blood_registration_number", "jockey_code", "trainer_code",
                "impost_weight", "base_win_odds", "idm", "jockey_index",
                "info_index", "training_index", "stable_index", "composite_index",
                "running_style_code", "distance_aptitude_code",
                "ten_index_pred", "pace_index_pred", "agari_index_pred",
                "position_index_pred", "pace_prediction",
                "improvement_grade", "upset_index", "longshot_index",
                "heavy_aptitude_code", "hoof_code", "blinkers",
                "turf_aptitude_code", "dirt_aptitude_code",
                "start_index", "delay_rate", "stable_rank",
            )

            if existing:
                existing.horse_id = horse_id or existing.horse_id
                existing.jockey_id = jockey_id or existing.jockey_id
                existing.trainer_id = trainer_id or existing.trainer_id
                for attr in _ENTRY_ATTRS:
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                kwargs = {
                    "race_id": race.id,
                    "post_position": post_pos,
                    "horse_id": horse_id,
                    "jockey_id": jockey_id,
                    "trainer_id": trainer_id,
                }
                for attr in _ENTRY_ATTRS:
                    kwargs[attr] = rec.get(attr)
                session.add(KeibaRaceEntry(**kwargs))
                count += 1
        session.flush()
        return count

    def save_results(self, session: Session, records: list[dict]) -> int:
        count = 0
        for rec in records:
            race_key = self._build_race_key(rec)
            if not race_key:
                continue
            race = session.execute(
                select(KeibaRace).where(KeibaRace.race_key == race_key)
            ).scalar_one_or_none()
            if race is None:
                continue

            post_pos = rec.get("post_position")
            if post_pos is None:
                continue

            entry = session.execute(
                select(KeibaRaceEntry).where(
                    KeibaRaceEntry.race_id == race.id,
                    KeibaRaceEntry.post_position == post_pos,
                )
            ).scalar_one_or_none()
            if entry is None:
                self._log.warning("entry_not_found", race_key=race_key, post_position=post_pos)
                continue

            existing = session.execute(
                select(KeibaRaceResult).where(KeibaRaceResult.entry_id == entry.id)
            ).scalar_one_or_none()

            _RESULT_ATTRS = (
                "finish_position", "race_time", "last_3f_time", "first_3f_time",
                "margin", "corner_1_pos", "corner_2_pos", "corner_3_pos", "corner_4_pos",
                "confirmed_win_odds", "confirmed_popularity",
                "jrdb_idm", "base_score", "track_bias", "pace_factor",
                "late_start_correction", "positioning_correction", "disadvantage_correction",
                "course_position", "ten_index", "agari_index", "pace_index",
                "body_weight", "body_weight_change",
            )

            if existing:
                for attr in _RESULT_ATTRS:
                    val = rec.get(attr)
                    if val is not None:
                        setattr(existing, attr, val)
            else:
                kwargs = {"entry_id": entry.id}
                for attr in _RESULT_ATTRS:
                    kwargs[attr] = rec.get(attr)
                session.add(KeibaRaceResult(**kwargs))
                count += 1
        session.flush()
        return count

    def save_payouts(self, session: Session, race_key: str, payouts: list[dict]) -> int:
        race = session.execute(
            select(KeibaRace).where(KeibaRace.race_key == race_key)
        ).scalar_one_or_none()
        if race is None:
            self._log.warning("race_not_found_for_payout", race_key=race_key)
            return 0
        count = 0
        for p in payouts:
            ticket_type = p["ticket_type"]
            combination = p["combination"]
            existing = session.execute(
                select(KeibaTicketPayout).where(
                    KeibaTicketPayout.race_id == race.id,
                    KeibaTicketPayout.ticket_type == ticket_type,
                    KeibaTicketPayout.combination == combination,
                )
            ).scalar_one_or_none()
            if existing:
                existing.payout_amount = p["payout_amount"]
            else:
                session.add(KeibaTicketPayout(
                    race_id=race.id,
                    ticket_type=ticket_type,
                    combination=combination,
                    payout_amount=p["payout_amount"],
                ))
                count += 1
        session.flush()
        return count

    def log_import(self, session: Session, file_name: str, file_type: str, record_count: int) -> None:
        existing = session.execute(
            select(KeibaImportLog).where(KeibaImportLog.file_name == file_name)
        ).scalar_one_or_none()
        if existing:
            existing.record_count = record_count
        else:
            session.add(KeibaImportLog(file_name=file_name, file_type=file_type, record_count=record_count))
        session.flush()

    def is_imported(self, session: Session, file_name: str) -> bool:
        return session.execute(
            select(KeibaImportLog).where(KeibaImportLog.file_name == file_name)
        ).scalar_one_or_none() is not None

    # --- helpers ---

    @staticmethod
    def _build_race_key(rec: dict) -> str | None:
        parts = [rec.get("place_code"), rec.get("year"), rec.get("kai")]
        day = rec.get("day")
        race_num = rec.get("race_number")
        if any(p is None for p in parts) or day is None or race_num is None:
            return None
        from providence.keiba.race_key import _INT_TO_HEX
        day_hex = _INT_TO_HEX.get(day, str(day)) if isinstance(day, int) else str(day)
        return f"{parts[0]}{parts[1]}{parts[2]}{day_hex}{race_num}"

    @staticmethod
    def _parse_date(date_str: str | None):
        if not date_str or len(date_str) != 8:
            return None
        from datetime import date as dt_date
        try:
            return dt_date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_racecourse_id(place_code: str | None) -> int | None:
        if not place_code:
            return None
        try:
            return int(place_code)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _resolve_horse_id(session: Session, brn: str | None) -> int | None:
        if not brn:
            return None
        horse = session.execute(
            select(KeibaHorse).where(KeibaHorse.blood_registration_number == brn)
        ).scalar_one_or_none()
        return horse.id if horse else None

    @staticmethod
    def _resolve_jockey_id(session: Session, code: str | None) -> int | None:
        if not code:
            return None
        jockey = session.execute(
            select(KeibaJockey).where(KeibaJockey.jockey_code == code)
        ).scalar_one_or_none()
        return jockey.id if jockey else None

    @staticmethod
    def _resolve_trainer_id(session: Session, code: str | None) -> int | None:
        if not code:
            return None
        trainer = session.execute(
            select(KeibaTrainer).where(KeibaTrainer.trainer_code == code)
        ).scalar_one_or_none()
        return trainer.id if trainer else None
