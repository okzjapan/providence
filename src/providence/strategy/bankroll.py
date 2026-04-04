"""Stake normalization: convert Kelly weights to JPY amounts."""

from __future__ import annotations

from providence.strategy.types import RecommendedBet, StrategyConfig


def normalize_to_stakes(
    recommendations: list[RecommendedBet],
    *,
    config: StrategyConfig,
) -> list[RecommendedBet]:
    """Normalize weights so the smallest qualifying bet equals ``min_bet_amount``.

    Steps:
    1. Filter out weights below ``min_weight_threshold`` (safety valve against
       a tiny weight inflating total stake).
    2. Scale so that the smallest surviving weight maps to ``min_bet_amount``
       (default 100 JPY), then round each bet down to the nearest
       ``min_bet_amount``.
    3. If ``max_total_stake`` is set and the total exceeds it, re-scale all
       bets proportionally and re-round.
    """
    if not recommendations:
        return []

    eligible = [r for r in recommendations if r.stake_weight >= config.min_weight_threshold]
    if not eligible:
        return []

    min_weight = min(r.stake_weight for r in eligible)
    if min_weight <= 0:
        return []

    scale = config.min_bet_amount / min_weight
    rounded = _round_bets(eligible, scale, config)
    if not rounded:
        return []

    total = sum(r.recommended_bet for r in rounded)
    if config.max_total_stake > 0 and total > config.max_total_stake:
        shrink = config.max_total_stake / total
        scale *= shrink
        rounded = _round_bets(eligible, scale, config)

    return rounded


def _round_bets(
    eligible: list[RecommendedBet],
    scale: float,
    config: StrategyConfig,
) -> list[RecommendedBet]:
    result: list[RecommendedBet] = []
    for rec in sorted(eligible, key=lambda r: r.stake_weight, reverse=True):
        raw = rec.stake_weight * scale
        rounded_bet = int(raw // config.min_bet_amount) * config.min_bet_amount
        if rounded_bet < config.min_bet_amount:
            continue
        result.append(
            RecommendedBet(
                ticket_type=rec.ticket_type,
                combination=rec.combination,
                probability=rec.probability,
                odds_value=rec.odds_value,
                expected_value=rec.expected_value,
                confidence_score=rec.confidence_score,
                kelly_fraction=rec.kelly_fraction,
                recommended_bet=float(rounded_bet),
                stake_weight=rec.stake_weight,
                skip_reason=rec.skip_reason,
            )
        )
    return result
