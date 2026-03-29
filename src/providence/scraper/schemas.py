"""Pydantic schemas for scraper output.

Both autorace.jp and oddspark scrapers convert their source-specific
responses into these common models before handing data to the repository layer.
"""

from datetime import date

from pydantic import BaseModel, Field

from providence.domain.enums import (
    EntryStatus,
    Grade,
    RiderRank,
    TicketType,
    TrackCode,
    TrackCondition,
)


class EntryRow(BaseModel):
    """1 rider entry in a race."""

    post_position: int = Field(ge=1, le=8)
    rider_registration_number: str
    rider_name: str
    age: int | None = None
    generation: int | None = None
    rank: RiderRank | None = None
    handicap_meters: int = Field(ge=0, le=110)
    trial_time: float | None = None
    avg_trial_time: float | None = None
    trial_deviation: float | None = None
    race_score: float | None = None


class RaceEntriesResponse(BaseModel):
    """Race entry list (common output for both sources)."""

    track: TrackCode
    race_date: date
    race_number: int = Field(ge=1, le=12)
    grade: Grade = Grade.NORMAL
    title: str | None = None
    distance: int = 3100
    weather: str | None = None
    track_condition: TrackCondition | None = None
    temperature: float | None = None
    humidity: float | None = None
    track_temperature: float | None = None
    entries: list[EntryRow]


class ResultRow(BaseModel):
    """1 rider result in a race."""

    post_position: int = Field(ge=1, le=8)
    rider_registration_number: str
    finish_position: int | None = None
    race_time: float | None = None
    trial_time: float | None = None
    start_timing: float | None = None
    accident_code: str | None = None
    entry_status: EntryStatus = EntryStatus.RACING


class RefundRow(BaseModel):
    """Refund info for a winning combination."""

    ticket_type: TicketType
    combination: str
    refund_amount: int = Field(ge=0)
    popularity: int | None = None


class RaceResultResponse(BaseModel):
    """Race result (common output for both sources)."""

    track: TrackCode
    race_date: date
    race_number: int = Field(ge=1, le=12)
    weather: str | None = None
    track_condition: TrackCondition | None = None
    temperature: float | None = None
    humidity: float | None = None
    track_temperature: float | None = None
    results: list[ResultRow]
    refunds: list[RefundRow]


class OddsRow(BaseModel):
    """One odds entry."""

    ticket_type: TicketType
    combination: str
    odds_value: float = Field(gt=0)
    popularity: int | None = None


class OddsResponse(BaseModel):
    """Odds for a race (common output for both sources)."""

    track: TrackCode
    race_date: date
    race_number: int = Field(ge=1, le=12)
    odds: list[OddsRow]


class PlayerSummary(BaseModel):
    """Player summary from ranking list."""

    registration_number: str
    name: str
    age: int | None = None
    generation: int | None = None
    rank: RiderRank | None = None
    home_track: TrackCode | None = None


class PlayerProfileResponse(BaseModel):
    """Detailed player profile."""

    registration_number: str
    name: str
    name_kana: str | None = None
    birth_year: int | None = None
    generation: int | None = None
    home_track: TrackCode | None = None
    rank: RiderRank | None = None
