"""Candidate selection from probabilities and market odds."""

from __future__ import annotations

from providence.strategy.expected_value import compute_expected_value
from providence.strategy.types import MarketTicketOdds, PredictedTicketProb, StrategyConfig, TicketCandidate


def build_candidates(
    predicted_probs: list[PredictedTicketProb],
    market_odds: list[MarketTicketOdds],
    *,
    confidence_score: float,
    config: StrategyConfig,
) -> list[TicketCandidate]:
    odds_lookup = {(row.ticket_type, row.combination): row for row in market_odds}
    candidates: list[TicketCandidate] = []

    for predicted in predicted_probs:
        odds = odds_lookup.get((predicted.ticket_type, predicted.combination))
        if odds is None:
            continue

        expected_value = compute_expected_value(predicted.probability, odds.odds_value)
        if expected_value < config.min_expected_value:
            continue
        if predicted.probability < config.min_probability:
            continue

        candidates.append(
            TicketCandidate(
                ticket_type=predicted.ticket_type,
                combination=predicted.combination,
                probability=predicted.probability,
                odds_value=odds.odds_value,
                expected_value=expected_value,
                confidence_score=confidence_score,
            )
        )

    candidates.sort(
        key=lambda row: (row.expected_value, row.probability, row.odds_value),
        reverse=True,
    )
    return candidates[: config.max_candidates]
