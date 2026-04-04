"""High-level strategy optimizer."""

from __future__ import annotations

from providence.strategy.bankroll import normalize_to_stakes
from providence.strategy.candidates import build_candidates
from providence.strategy.confidence import race_confidence
from providence.strategy.kelly import optimize_kelly_fractions
from providence.strategy.normalize import flatten_ticket_probs
from providence.strategy.types import (
    DecisionContext,
    MarketTicketOdds,
    RacePredictionBundle,
    RecommendedBet,
    StrategyConfig,
    StrategyRunResult,
)


def run_strategy(
    bundle: RacePredictionBundle,
    market_odds: list[MarketTicketOdds],
    *,
    decision_context: DecisionContext,
    config: StrategyConfig | None = None,
) -> StrategyRunResult:
    config = config or StrategyConfig()
    confidence_score = race_confidence(bundle)
    if not market_odds:
        return StrategyRunResult(
            race_id=bundle.race_id,
            model_version=bundle.model_version,
            decision_context=decision_context,
            confidence_score=confidence_score,
            skip_reason="no_market_odds",
        )
    if confidence_score < config.min_confidence:
        return StrategyRunResult(
            race_id=bundle.race_id,
            model_version=bundle.model_version,
            decision_context=decision_context,
            confidence_score=confidence_score,
            skip_reason="low_confidence",
        )

    predicted = flatten_ticket_probs(bundle.ticket_probs, bundle.index_map)
    candidates = build_candidates(predicted, market_odds, confidence_score=confidence_score, config=config)
    if not candidates:
        return StrategyRunResult(
            race_id=bundle.race_id,
            model_version=bundle.model_version,
            decision_context=decision_context,
            confidence_score=confidence_score,
            skip_reason="no_positive_ev_candidates",
        )

    weights = optimize_kelly_fractions(
        candidates=candidates,
        bundle=bundle,
    )
    weights *= config.fractional_kelly

    recommendations = [
        RecommendedBet(
            ticket_type=candidate.ticket_type,
            combination=candidate.combination,
            probability=candidate.probability,
            odds_value=candidate.odds_value,
            expected_value=candidate.expected_value,
            confidence_score=candidate.confidence_score,
            kelly_fraction=float(weight),
            recommended_bet=0.0,
            stake_weight=float(weight),
        )
        for candidate, weight in zip(candidates, weights, strict=False)
        if weight > 0
    ]

    rounded = normalize_to_stakes(recommendations, config=config)
    skip_reason = None if rounded else "rounded_below_minimum"
    return StrategyRunResult(
        race_id=bundle.race_id,
        model_version=bundle.model_version,
        decision_context=decision_context,
        confidence_score=confidence_score,
        candidate_bets=recommendations,
        recommended_bets=rounded,
        skip_reason=skip_reason,
    )
