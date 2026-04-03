"""Settlement helpers for strategy recommendations."""

from __future__ import annotations

from providence.strategy.types import RecommendedBet, SettledTicketPayout

from .types import SettledRecommendation


def settle_recommendations(
    recommendations: list[RecommendedBet],
    payouts: list[SettledTicketPayout],
) -> list[SettledRecommendation]:
    payout_lookup = {(row.ticket_type, row.combination): row for row in payouts}
    settled: list[SettledRecommendation] = []
    for recommendation in recommendations:
        payout = payout_lookup.get((recommendation.ticket_type, recommendation.combination))
        payout_amount = recommendation.recommended_bet * payout.payout_value if payout else 0.0
        profit = payout_amount - recommendation.recommended_bet
        settled.append(
            SettledRecommendation(
                recommendation=recommendation,
                payout_amount=float(payout_amount),
                profit=float(profit),
                hit=payout is not None,
            )
        )
    return settled
