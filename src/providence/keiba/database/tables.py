"""Keiba (JRA horse racing) database table definitions.

All tables use the shared Base from providence.database.tables.
Tables are prefixed with 'keiba_' to avoid collisions with autorace tables.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from providence.database.tables import Base


class KeibaRacecourse(Base):
    __tablename__ = "keiba_racecourses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    name_en: Mapped[str | None] = mapped_column(String)

    races: Mapped[list["KeibaRace"]] = relationship(back_populates="racecourse")


class KeibaHorse(Base):
    __tablename__ = "keiba_horses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    blood_registration_number: Mapped[str] = mapped_column(String(8), nullable=False, unique=True)
    horse_name: Mapped[str | None] = mapped_column(String)
    sex_code: Mapped[int | None] = mapped_column(Integer)
    birth_date: Mapped[str | None] = mapped_column(String(8))
    sire_name: Mapped[str | None] = mapped_column(String)
    dam_name: Mapped[str | None] = mapped_column(String)
    broodmare_sire_name: Mapped[str | None] = mapped_column(String)
    sire_code: Mapped[str | None] = mapped_column(String(4))
    broodmare_sire_code: Mapped[str | None] = mapped_column(String(4))
    retired_flag: Mapped[int | None] = mapped_column(Integer)

    sire_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_horses.id"))
    dam_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_horses.id"))
    broodmare_sire_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_horses.id"))

    entries: Mapped[list["KeibaRaceEntry"]] = relationship(back_populates="horse")


class KeibaJockey(Base):
    __tablename__ = "keiba_jockeys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    jockey_code: Mapped[str] = mapped_column(String(5), nullable=False, unique=True)
    jockey_name: Mapped[str | None] = mapped_column(String)
    affiliation_code: Mapped[int | None] = mapped_column(Integer)
    apprentice_class: Mapped[int | None] = mapped_column(Integer)
    retired_flag: Mapped[int | None] = mapped_column(Integer)
    career_flat_1st: Mapped[int | None] = mapped_column(Integer)
    career_flat_2nd: Mapped[int | None] = mapped_column(Integer)
    career_flat_3rd: Mapped[int | None] = mapped_column(Integer)
    career_flat_unplaced: Mapped[int | None] = mapped_column(Integer)

    entries: Mapped[list["KeibaRaceEntry"]] = relationship(back_populates="jockey")


class KeibaTrainer(Base):
    __tablename__ = "keiba_trainers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trainer_code: Mapped[str] = mapped_column(String(5), nullable=False, unique=True)
    trainer_name: Mapped[str | None] = mapped_column(String)
    affiliation_code: Mapped[int | None] = mapped_column(Integer)
    retired_flag: Mapped[int | None] = mapped_column(Integer)
    career_flat_1st: Mapped[int | None] = mapped_column(Integer)
    career_flat_2nd: Mapped[int | None] = mapped_column(Integer)
    career_flat_3rd: Mapped[int | None] = mapped_column(Integer)
    career_flat_unplaced: Mapped[int | None] = mapped_column(Integer)

    entries: Mapped[list["KeibaRaceEntry"]] = relationship(back_populates="trainer")


class KeibaRace(Base):
    __tablename__ = "keiba_races"
    __table_args__ = (
        UniqueConstraint("race_key", name="uq_keiba_races_key"),
        Index("ix_keiba_races_date", "race_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_key: Mapped[str] = mapped_column(String(8), nullable=False)
    race_date: Mapped[date | None] = mapped_column(Date)
    racecourse_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_racecourses.id"))
    race_number: Mapped[int | None] = mapped_column(Integer)
    distance: Mapped[int | None] = mapped_column(Integer)
    surface_code: Mapped[int | None] = mapped_column(Integer)
    direction_code: Mapped[int | None] = mapped_column(Integer)
    course_type_code: Mapped[int | None] = mapped_column(Integer)
    going_code: Mapped[int | None] = mapped_column(Integer)
    weather_code: Mapped[int | None] = mapped_column(Integer)
    race_name: Mapped[str | None] = mapped_column(String)
    class_code: Mapped[str | None] = mapped_column(String)
    weight_rule_code: Mapped[str | None] = mapped_column(String)
    num_runners: Mapped[int | None] = mapped_column(Integer)
    post_time: Mapped[str | None] = mapped_column(String(4))

    racecourse: Mapped[KeibaRacecourse | None] = relationship(back_populates="races")
    entries: Mapped[list["KeibaRaceEntry"]] = relationship(back_populates="race", cascade="all, delete-orphan")
    payouts: Mapped[list["KeibaTicketPayout"]] = relationship(back_populates="race", cascade="all, delete-orphan")


class KeibaRaceEntry(Base):
    __tablename__ = "keiba_race_entries"
    __table_args__ = (
        UniqueConstraint("race_id", "post_position", name="uq_keiba_entry_identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("keiba_races.id"), nullable=False)
    post_position: Mapped[int] = mapped_column(Integer, nullable=False)
    horse_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_horses.id"))
    jockey_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_jockeys.id"))
    trainer_id: Mapped[int | None] = mapped_column(ForeignKey("keiba_trainers.id"))
    blood_registration_number: Mapped[str | None] = mapped_column(String(8))
    jockey_code: Mapped[str | None] = mapped_column(String(5))
    trainer_code: Mapped[str | None] = mapped_column(String(5))
    impost_weight: Mapped[float | None] = mapped_column(Float)
    base_win_odds: Mapped[float | None] = mapped_column(Float)
    idm: Mapped[float | None] = mapped_column(Float)
    jockey_index: Mapped[float | None] = mapped_column(Float)
    info_index: Mapped[float | None] = mapped_column(Float)
    training_index: Mapped[float | None] = mapped_column(Float)
    stable_index: Mapped[float | None] = mapped_column(Float)
    composite_index: Mapped[float | None] = mapped_column(Float)
    running_style_code: Mapped[int | None] = mapped_column(Integer)
    distance_aptitude_code: Mapped[str | None] = mapped_column(String)

    race: Mapped[KeibaRace] = relationship(back_populates="entries")
    horse: Mapped[KeibaHorse | None] = relationship(back_populates="entries")
    jockey: Mapped[KeibaJockey | None] = relationship(back_populates="entries")
    trainer: Mapped[KeibaTrainer | None] = relationship(back_populates="entries")
    result: Mapped["KeibaRaceResult | None"] = relationship(back_populates="entry", uselist=False)


class KeibaRaceResult(Base):
    __tablename__ = "keiba_race_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("keiba_race_entries.id"), nullable=False, unique=True)
    finish_position: Mapped[int | None] = mapped_column(Integer)
    race_time: Mapped[float | None] = mapped_column(Float)
    last_3f_time: Mapped[float | None] = mapped_column(Float)
    first_3f_time: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    corner_1_pos: Mapped[int | None] = mapped_column(Integer)
    corner_2_pos: Mapped[int | None] = mapped_column(Integer)
    corner_3_pos: Mapped[int | None] = mapped_column(Integer)
    corner_4_pos: Mapped[int | None] = mapped_column(Integer)
    confirmed_win_odds: Mapped[float | None] = mapped_column(Float)
    confirmed_popularity: Mapped[int | None] = mapped_column(Integer)
    jrdb_speed_figure: Mapped[float | None] = mapped_column(Float)
    body_weight: Mapped[int | None] = mapped_column(Integer)
    body_weight_change: Mapped[int | None] = mapped_column(Integer)

    entry: Mapped[KeibaRaceEntry] = relationship(back_populates="result")


class KeibaTicketPayout(Base):
    __tablename__ = "keiba_ticket_payouts"
    __table_args__ = (
        UniqueConstraint("race_id", "ticket_type", "combination", name="uq_keiba_payout_identity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("keiba_races.id"), nullable=False)
    ticket_type: Mapped[str] = mapped_column(String, nullable=False)
    combination: Mapped[str] = mapped_column(String, nullable=False)
    payout_amount: Mapped[int] = mapped_column(Integer, nullable=False)

    race: Mapped[KeibaRace] = relationship(back_populates="payouts")


class KeibaImportLog(Base):
    __tablename__ = "keiba_import_log"
    __table_args__ = (
        UniqueConstraint("file_name", name="uq_keiba_import_log_file"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    record_count: Mapped[int | None] = mapped_column(Integer)
    imported_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
