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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)

    races: Mapped[list["Race"]] = relationship(back_populates="track")
    riders: Mapped[list["Rider"]] = relationship(back_populates="home_track")


class Rider(Base):
    __tablename__ = "riders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    registration_number: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    name_kana: Mapped[str | None] = mapped_column(String, nullable=True)
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_track_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("tracks.id"), nullable=True)
    generation: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_track: Mapped[Track | None] = relationship(back_populates="riders")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="rider")


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("track_id", "race_date", "race_number", name="uq_race_identity"),
        Index("ix_races_date", "race_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(Integer, ForeignKey("tracks.id"), nullable=False)
    race_date: Mapped[date] = mapped_column(Date, nullable=False)
    race_number: Mapped[int] = mapped_column(Integer, nullable=False)
    grade: Mapped[str] = mapped_column(String, nullable=False, default="普通")
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    distance: Mapped[int] = mapped_column(Integer, nullable=False, default=3100)
    track_condition: Mapped[str | None] = mapped_column(String, nullable=True)
    weather: Mapped[str | None] = mapped_column(String, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity: Mapped[float | None] = mapped_column(Float, nullable=True)
    track_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="正常")

    track: Mapped[Track] = relationship(back_populates="races")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="race", cascade="all, delete-orphan")
    odds_snapshots: Mapped[list["OddsSnapshot"]] = relationship(back_populates="race", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="race", cascade="all, delete-orphan")


class RaceEntry(Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        UniqueConstraint("race_id", "post_position", name="uq_entry_position"),
        Index("ix_race_entries_rider", "rider_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), nullable=False)
    rider_id: Mapped[int] = mapped_column(Integer, ForeignKey("riders.id"), nullable=False)
    post_position: Mapped[int] = mapped_column(Integer, nullable=False)
    handicap_meters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trial_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_trial_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    trial_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    race_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_status: Mapped[str] = mapped_column(String, nullable=False, default="出走")

    race: Mapped[Race] = relationship(back_populates="entries")
    rider: Mapped[Rider] = relationship(back_populates="entries")
    result: Mapped["RaceResult | None"] = relationship(
        back_populates="entry", uselist=False, cascade="all, delete-orphan"
    )


class RaceResult(Base):
    __tablename__ = "race_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_entry_id: Mapped[int] = mapped_column(Integer, ForeignKey("race_entries.id"), unique=True, nullable=False)
    finish_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    race_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_timing: Mapped[float | None] = mapped_column(Float, nullable=True)
    accident_code: Mapped[str | None] = mapped_column(String, nullable=True)

    entry: Mapped[RaceEntry] = relationship(back_populates="result")


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshot"
    __table_args__ = (Index("ix_odds_race_type", "race_id", "ticket_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), nullable=False)
    ticket_type: Mapped[str] = mapped_column(String, nullable=False)
    combination: Mapped[str] = mapped_column(String, nullable=False)
    odds_value: Mapped[float] = mapped_column(Float, nullable=False)
    popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    race: Mapped[Race] = relationship(back_populates="odds_snapshots")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    ticket_type: Mapped[str] = mapped_column(String, nullable=False)
    combination: Mapped[str] = mapped_column(String, nullable=False)
    predicted_prob: Mapped[float] = mapped_column(Float, nullable=False)
    odds_at_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    kelly_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_bet: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    race: Mapped[Race] = relationship(back_populates="predictions")
    betting_log: Mapped["BettingLog | None"] = relationship(back_populates="prediction", uselist=False)


class BettingLog(Base):
    __tablename__ = "betting_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    race_id: Mapped[int] = mapped_column(Integer, ForeignKey("races.id"), nullable=False)
    actual_bet_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payout: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    profit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bankroll_after: Mapped[float | None] = mapped_column(Float, nullable=True)

    prediction: Mapped[Prediction] = relationship(back_populates="betting_log")


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    evaluation_date: Mapped[date] = mapped_column(Date, nullable=False)
    window: Mapped[str] = mapped_column(String, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    win_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    top3_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)


class ScrapeLog(Base):
    __tablename__ = "scrape_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    source: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    target_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    records_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
