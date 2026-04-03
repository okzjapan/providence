"""Core strategy dataclasses shared by live prediction and backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from providence.domain.enums import TicketType


class EvaluationMode(StrEnum):
    LIVE = "live"
    FIXED = "fixed"
    WALK_FORWARD = "walk-forward"


@dataclass(frozen=True)
class DecisionContext:
    judgment_time: datetime
    evaluation_mode: EvaluationMode = EvaluationMode.LIVE
    timezone: str = "UTC"
    provenance: str = "cli"


@dataclass(frozen=True)
class RaceIndexMap:
    index_to_post_position: tuple[int, ...]
    index_to_entry_id: tuple[int, ...]

    def post_position_for_index(self, index: int) -> int:
        return self.index_to_post_position[index]

    def entry_id_for_index(self, index: int) -> int:
        return self.index_to_entry_id[index]

    def index_for_post_position(self, post_position: int) -> int:
        try:
            return self.index_to_post_position.index(post_position)
        except ValueError as exc:
            raise KeyError(f"Unknown post_position: {post_position}") from exc


@dataclass(frozen=True)
class PredictedTicketProb:
    ticket_type: TicketType
    combination: tuple[int, ...]
    probability: float


@dataclass(frozen=True)
class MarketTicketOdds:
    ticket_type: TicketType
    combination: tuple[int, ...]
    odds_value: float
    captured_at: datetime
    ingestion_batch_id: str
    source_name: str | None = None


@dataclass(frozen=True)
class SettledTicketPayout:
    ticket_type: TicketType
    combination: tuple[int, ...]
    payout_value: float
    settled_at: datetime


@dataclass(frozen=True)
class TicketCandidate:
    ticket_type: TicketType
    combination: tuple[int, ...]
    probability: float
    odds_value: float
    expected_value: float
    confidence_score: float


@dataclass(frozen=True)
class RecommendedBet:
    ticket_type: TicketType
    combination: tuple[int, ...]
    probability: float
    odds_value: float
    expected_value: float
    confidence_score: float
    kelly_fraction: float
    recommended_bet: float
    skip_reason: str | None = None


@dataclass(frozen=True)
class RacePredictionBundle:
    race_id: int
    model_version: str
    temperature: float
    scores: tuple[float, ...]
    index_map: RaceIndexMap
    ticket_probs: dict[str, dict]
    features_total_races: tuple[int, ...]


@dataclass(frozen=True)
class StrategyConfig:
    fractional_kelly: float = 0.25
    race_cap_fraction: float = 0.05
    daily_loss_limit_fraction: float = 0.10
    min_bet_amount: int = 100
    min_expected_value: float = 0.0
    min_probability: float = 0.0
    max_candidates: int = 12
    min_confidence: float = 0.1


@dataclass
class StrategyRunResult:
    race_id: int
    model_version: str
    decision_context: DecisionContext
    confidence_score: float
    candidate_bets: list[RecommendedBet] = field(default_factory=list)
    recommended_bets: list[RecommendedBet] = field(default_factory=list)
    skip_reason: str | None = None
    bankroll_before: float | None = None
    bankroll_after: float | None = None

    @property
    def total_recommended_bet(self) -> float:
        return float(sum(b.recommended_bet for b in self.recommended_bets))


def utcnow() -> datetime:
    return datetime.now(UTC)
