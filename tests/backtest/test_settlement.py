from datetime import datetime

from providence.domain.enums import TicketType
from providence.strategy.types import RecommendedBet, SettledTicketPayout

from providence.backtest.settlement import settle_recommendations


def test_settlement_uses_payout_multiplier():
    recommendation = RecommendedBet(
        ticket_type=TicketType.WIN,
        combination=(1,),
        probability=0.4,
        odds_value=3.0,
        expected_value=0.2,
        confidence_score=0.8,
        kelly_fraction=0.03,
        recommended_bet=200.0,
    )
    payout = SettledTicketPayout(
        ticket_type=TicketType.WIN,
        combination=(1,),
        payout_value=3.5,
        settled_at=datetime(2025, 6, 15, 12, 0, 0),
    )
    settled = settle_recommendations([recommendation], [payout])
    assert settled[0].hit is True
    assert settled[0].payout_amount == 700.0
    assert settled[0].profit == 500.0
