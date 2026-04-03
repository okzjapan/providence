"""Normalization helpers between model outputs, DB rows, and strategy types."""

from __future__ import annotations

from datetime import UTC, datetime

from providence.database.tables import OddsSnapshot, TicketPayout
from providence.domain.enums import TicketType
from providence.strategy.types import MarketTicketOdds, PredictedTicketProb, RaceIndexMap, SettledTicketPayout

_KEY_TO_TICKET_TYPE: dict[str, TicketType] = {
    "win": TicketType.WIN,
    "exacta": TicketType.EXACTA,
    "quinella": TicketType.QUINELLA,
    "wide": TicketType.WIDE,
    "trifecta": TicketType.TRIFECTA,
    "trio": TicketType.TRIO,
}


def ticket_type_for_key(key: str) -> TicketType:
    try:
        return _KEY_TO_TICKET_TYPE[key]
    except KeyError as exc:
        raise KeyError(f"Unsupported probability key: {key}") from exc


def is_ordered_ticket_type(ticket_type: TicketType) -> bool:
    return ticket_type in {TicketType.WIN, TicketType.EXACTA, TicketType.TRIFECTA}


def to_post_position_combination(
    combination: int | tuple[int, ...],
    index_map: RaceIndexMap,
) -> tuple[int, ...]:
    if isinstance(combination, int):
        return (index_map.post_position_for_index(combination),)
    return tuple(index_map.post_position_for_index(idx) for idx in combination)


def combination_to_indices(combination: tuple[int, ...], index_map: RaceIndexMap) -> tuple[int, ...]:
    return tuple(index_map.index_for_post_position(value) for value in combination)


def format_combination(ticket_type: TicketType, combination: tuple[int, ...]) -> str:
    values = combination if is_ordered_ticket_type(ticket_type) else tuple(sorted(combination))
    return "-".join(str(value) for value in values)


def parse_combination(ticket_type: TicketType, combination: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in combination.split("-"))
    return values if is_ordered_ticket_type(ticket_type) else tuple(sorted(values))


def flatten_ticket_probs(ticket_probs: dict[str, dict], index_map: RaceIndexMap) -> list[PredictedTicketProb]:
    flattened: list[PredictedTicketProb] = []
    for key, values in ticket_probs.items():
        ticket_type = ticket_type_for_key(key)
        for combo, probability in values.items():
            combination = to_post_position_combination(combo, index_map)
            flattened.append(
                PredictedTicketProb(
                    ticket_type=ticket_type,
                    combination=combination,
                    probability=float(probability),
                )
            )
    return flattened


def market_odds_from_rows(rows: list[OddsSnapshot]) -> list[MarketTicketOdds]:
    market_odds: list[MarketTicketOdds] = []
    for row in rows:
        if row.ingestion_batch_id is None:
            continue
        ticket_type = TicketType(row.ticket_type)
        market_odds.append(
            MarketTicketOdds(
                ticket_type=ticket_type,
                combination=parse_combination(ticket_type, row.combination),
                odds_value=float(row.odds_value),
                captured_at=_as_utc(row.captured_at),
                ingestion_batch_id=row.ingestion_batch_id,
                source_name=row.source_name,
            )
        )
    return market_odds


def payouts_from_rows(rows: list[TicketPayout]) -> list[SettledTicketPayout]:
    payouts: list[SettledTicketPayout] = []
    for row in rows:
        ticket_type = TicketType(row.ticket_type)
        payouts.append(
            SettledTicketPayout(
                ticket_type=ticket_type,
                combination=parse_combination(ticket_type, row.combination),
                payout_value=float(row.payout_value),
                settled_at=_as_utc(row.settled_at),
            )
        )
    return payouts


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    return value.replace(tzinfo=UTC)
