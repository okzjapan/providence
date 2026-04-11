"""Candidate selection from probabilities and market odds."""

from __future__ import annotations

from providence.strategy.expected_value import compute_expected_value
from providence.strategy.types import PAYOUT_RATE, MarketTicketOdds, PredictedTicketProb, StrategyConfig, TicketCandidate


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
        if config.allowed_ticket_types is not None and predicted.ticket_type not in config.allowed_ticket_types:
            continue
        odds = odds_lookup.get((predicted.ticket_type, predicted.combination))
        if odds is None:
            continue
        if config.min_odds is not None and odds.odds_value < config.min_odds:
            continue
        if config.max_odds is not None and odds.odds_value > config.max_odds:
            continue

        prob = predicted.probability
        if config.prob_power is not None and config.prob_power != 1.0:
            prob = prob ** config.prob_power

        expected_value = compute_expected_value(prob, odds.odds_value)
        if expected_value < config.min_expected_value:
            continue
        if prob < config.effective_min_probability(predicted.ticket_type):
            continue

        if config.min_edge is not None:
            market_implied = PAYOUT_RATE / odds.odds_value
            edge = prob - market_implied
            if edge < config.min_edge:
                continue

        candidates.append(
            TicketCandidate(
                ticket_type=predicted.ticket_type,
                combination=predicted.combination,
                probability=prob,
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
