from providence.domain.enums import TicketType
from providence.strategy.normalize import combination_to_indices, format_combination, parse_combination, to_post_position_combination
from providence.strategy.types import RaceIndexMap


def test_race_index_map_roundtrip_with_missing_post_positions():
    index_map = RaceIndexMap(
        index_to_post_position=(1, 3, 5, 8),
        index_to_entry_id=(10, 11, 12, 13),
    )

    combination = to_post_position_combination((0, 2), index_map)
    assert combination == (1, 5)
    assert combination_to_indices(combination, index_map) == (0, 2)


def test_format_combination_sorts_unordered_ticket_types():
    assert format_combination(TicketType.QUINELLA, (5, 1)) == "1-5"
    assert parse_combination(TicketType.QUINELLA, "5-1") == (1, 5)


def test_format_combination_preserves_ordered_ticket_types():
    assert format_combination(TicketType.EXACTA, (5, 1)) == "5-1"
    assert parse_combination(TicketType.EXACTA, "5-1") == (5, 1)
