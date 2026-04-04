import pytest

from providence.cli.strategy_options import build_strategy_config, parse_ticket_types
from providence.domain.enums import TicketType


class TestParseTicketTypes:
    def test_english_aliases(self):
        result = parse_ticket_types("win,wide")
        assert result == frozenset({TicketType.WIN, TicketType.WIDE})

    def test_japanese_names(self):
        result = parse_ticket_types("単勝,ワイド")
        assert result == frozenset({TicketType.WIN, TicketType.WIDE})

    def test_mixed_aliases(self):
        result = parse_ticket_types("win,ワイド,exacta")
        assert result == frozenset({TicketType.WIN, TicketType.WIDE, TicketType.EXACTA})

    def test_all_english_types(self):
        result = parse_ticket_types("win,exacta,quinella,wide,trifecta,trio")
        assert len(result) == 6

    def test_case_insensitive_english(self):
        result = parse_ticket_types("Win,WIDE")
        assert result == frozenset({TicketType.WIN, TicketType.WIDE})

    def test_whitespace_tolerance(self):
        result = parse_ticket_types("win , wide")
        assert result == frozenset({TicketType.WIN, TicketType.WIDE})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown ticket type"):
            parse_ticket_types("win,unknown")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            parse_ticket_types("")


class TestBuildStrategyConfig:
    def test_no_overrides_returns_defaults(self):
        config = build_strategy_config()
        assert config.allowed_ticket_types is None
        assert config.max_candidates == 12
        assert config.fractional_kelly == 0.25

    def test_ticket_types_override(self):
        config = build_strategy_config(ticket_types="win,wide")
        assert config.allowed_ticket_types == frozenset({TicketType.WIN, TicketType.WIDE})

    def test_max_candidates_override(self):
        config = build_strategy_config(max_candidates=24)
        assert config.max_candidates == 24

    def test_fractional_kelly_override(self):
        config = build_strategy_config(fractional_kelly=0.5)
        assert config.fractional_kelly == 0.5

    def test_combined_overrides(self):
        config = build_strategy_config(ticket_types="win,wide", max_candidates=24, fractional_kelly=0.5)
        assert config.allowed_ticket_types == frozenset({TicketType.WIN, TicketType.WIDE})
        assert config.max_candidates == 24
        assert config.fractional_kelly == 0.5

    def test_none_overrides_keep_defaults(self):
        config = build_strategy_config(ticket_types=None, max_candidates=None, fractional_kelly=None)
        assert config.allowed_ticket_types is None
        assert config.max_candidates == 12
        assert config.fractional_kelly == 0.25
