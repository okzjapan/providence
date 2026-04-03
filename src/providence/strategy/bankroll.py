"""Bankroll and rounding utilities."""

from __future__ import annotations

from providence.strategy.types import RecommendedBet, StrategyConfig


def race_budget(bankroll: float, config: StrategyConfig) -> float:
    return max(bankroll * config.race_cap_fraction, 0.0)


def round_recommended_bets(
    recommendations: list[RecommendedBet],
    *,
    bankroll: float,
    config: StrategyConfig,
) -> list[RecommendedBet]:
    if not recommendations:
        return []

    budget = race_budget(bankroll, config)
    rounded: list[RecommendedBet] = []
    spent = 0.0
    for recommendation in sorted(recommendations, key=lambda item: item.kelly_fraction, reverse=True):
        raw_bet = min(recommendation.recommended_bet, max(budget - spent, 0.0))
        rounded_bet = int(raw_bet // config.min_bet_amount) * config.min_bet_amount
        if rounded_bet < config.min_bet_amount:
            continue
        spent += rounded_bet
        rounded.append(
            RecommendedBet(
                ticket_type=recommendation.ticket_type,
                combination=recommendation.combination,
                probability=recommendation.probability,
                odds_value=recommendation.odds_value,
                expected_value=recommendation.expected_value,
                confidence_score=recommendation.confidence_score,
                kelly_fraction=recommendation.kelly_fraction,
                recommended_bet=float(rounded_bet),
                skip_reason=recommendation.skip_reason,
            )
        )
    return rounded
