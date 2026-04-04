from datetime import datetime

from providence.domain.enums import TicketType
from providence.strategy.optimizer import run_strategy
from providence.strategy.types import (
    DecisionContext,
    EvaluationMode,
    MarketTicketOdds,
    RaceIndexMap,
    RacePredictionBundle,
    StrategyConfig,
)


def _bundle() -> RacePredictionBundle:
    return RacePredictionBundle(
        race_id=1,
        model_version="v001",
        temperature=1.0,
        scores=(2.0, 1.0, 0.1),
        index_map=RaceIndexMap(index_to_post_position=(1, 3, 5), index_to_entry_id=(101, 102, 103)),
        ticket_probs={
            "win": {0: 0.6, 1: 0.3, 2: 0.1},
            "exacta": {(0, 1): 0.35, (1, 0): 0.1, (0, 2): 0.15},
            "quinella": {(0, 1): 0.45, (0, 2): 0.2, (1, 2): 0.15},
            "wide": {(0, 1): 0.7, (0, 2): 0.5, (1, 2): 0.3},
            "trifecta": {(0, 1, 2): 0.2},
            "trio": {(0, 1, 2): 0.4},
        },
        features_total_races=(12, 9, 7),
    )


def _context() -> DecisionContext:
    return DecisionContext(
        judgment_time=datetime(2025, 6, 15, 10, 0, 0),
        evaluation_mode=EvaluationMode.FIXED,
        provenance="test",
    )


def _market_odds() -> list[MarketTicketOdds]:
    return [
        MarketTicketOdds(TicketType.WIN, (1,), 2.4, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.EXACTA, (1, 3), 5.0, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.QUINELLA, (1, 3), 3.2, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.WIDE, (1, 3), 1.9, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.TRIFECTA, (1, 3, 5), 12.0, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.TRIO, (1, 3, 5), 4.0, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
    ]


def test_run_strategy_skips_without_market_odds():
    result = run_strategy(_bundle(), [], decision_context=_context())
    assert result.skip_reason == "no_market_odds"
    assert result.recommended_bets == []


def test_run_strategy_min_bet_is_100_and_all_multiples():
    config = StrategyConfig(max_candidates=6, min_confidence=0.0)
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    assert result.recommended_bets
    bets = [row.recommended_bet for row in result.recommended_bets]
    assert min(bets) == 100
    assert all(bet % 100 == 0 for bet in bets)


def test_run_strategy_respects_max_total_stake():
    config = StrategyConfig(max_candidates=6, min_confidence=0.0, max_total_stake=500)
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    assert sum(row.recommended_bet for row in result.recommended_bets) <= 500
    assert all(row.recommended_bet % 100 == 0 for row in result.recommended_bets)


def test_run_strategy_keeps_candidate_bets_when_rounded_to_zero():
    market_odds = [
        MarketTicketOdds(TicketType.WIN, (1,), 1.8, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.EXACTA, (1, 3), 2.1, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.QUINELLA, (1, 3), 2.0, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.WIDE, (1, 3), 1.5, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.TRIFECTA, (1, 3, 5), 3.0, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
        MarketTicketOdds(TicketType.TRIO, (1, 3, 5), 2.5, datetime(2025, 6, 15, 9, 50, 0), "batch-1"),
    ]
    config = StrategyConfig(max_candidates=6, min_confidence=0.0, min_weight_threshold=1.0)
    result = run_strategy(_bundle(), market_odds, decision_context=_context(), config=config)
    assert result.skip_reason == "rounded_below_minimum"
    assert result.recommended_bets == []
    assert result.candidate_bets


def test_run_strategy_stake_weight_is_positive():
    config = StrategyConfig(max_candidates=6, min_confidence=0.0)
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    for bet in result.recommended_bets:
        assert bet.stake_weight > 0


def test_run_strategy_tiny_weight_does_not_explode_total():
    """Ensure a near-zero weight doesn't cause total stake to blow up."""
    config = StrategyConfig(
        max_candidates=6,
        min_confidence=0.0,
        min_weight_threshold=0.001,
        max_total_stake=5_000,
    )
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    if result.recommended_bets:
        assert sum(row.recommended_bet for row in result.recommended_bets) <= 5_000


def test_allowed_ticket_types_filters_candidates():
    config = StrategyConfig(
        max_candidates=6,
        min_confidence=0.0,
        allowed_ticket_types=frozenset({TicketType.WIN, TicketType.WIDE}),
    )
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    all_types = {bet.ticket_type for bet in (result.recommended_bets or result.candidate_bets)}
    assert all_types <= {TicketType.WIN, TicketType.WIDE}


def test_allowed_ticket_types_none_allows_all():
    config = StrategyConfig(max_candidates=6, min_confidence=0.0, allowed_ticket_types=None)
    result = run_strategy(_bundle(), _market_odds(), decision_context=_context(), config=config)
    all_types = {bet.ticket_type for bet in (result.recommended_bets or result.candidate_bets)}
    assert len(all_types) > 2
