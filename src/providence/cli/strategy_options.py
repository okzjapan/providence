"""Shared CLI helpers for building StrategyConfig from command-line options."""

from __future__ import annotations

from providence.domain.enums import TicketType
from providence.strategy.types import StrategyConfig

_TICKET_TYPE_ALIASES: dict[str, TicketType] = {
    "win": TicketType.WIN,
    "単勝": TicketType.WIN,
    "place": TicketType.PLACE,
    "複勝": TicketType.PLACE,
    "exacta": TicketType.EXACTA,
    "2連単": TicketType.EXACTA,
    "quinella": TicketType.QUINELLA,
    "2連複": TicketType.QUINELLA,
    "wide": TicketType.WIDE,
    "ワイド": TicketType.WIDE,
    "trifecta": TicketType.TRIFECTA,
    "3連単": TicketType.TRIFECTA,
    "trio": TicketType.TRIO,
    "3連複": TicketType.TRIO,
}


def parse_ticket_types(raw: str) -> frozenset[TicketType]:
    """Parse a comma-separated string of ticket type names into a frozenset.

    Accepts both English aliases (``win``, ``wide``) and Japanese values
    (``単勝``, ``ワイド``).  Raises ``ValueError`` for unknown names.
    """
    parsed: set[TicketType] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        ticket_type = _TICKET_TYPE_ALIASES.get(token.lower()) or _TICKET_TYPE_ALIASES.get(token)
        if ticket_type is None:
            valid = sorted({k for k in _TICKET_TYPE_ALIASES if k.isascii()})
            raise ValueError(f"Unknown ticket type: '{token}'. Valid names: {', '.join(valid)}")
        parsed.add(ticket_type)
    if not parsed:
        raise ValueError("At least one ticket type is required")
    return frozenset(parsed)


def build_strategy_config(
    *,
    ticket_types: str | None = None,
    max_candidates: int | None = None,
    fractional_kelly: float | None = None,
) -> StrategyConfig:
    """Build a ``StrategyConfig`` from optional CLI overrides.

    Unspecified parameters fall back to the dataclass defaults.
    """
    overrides: dict[str, object] = {}
    if ticket_types is not None:
        overrides["allowed_ticket_types"] = parse_ticket_types(ticket_types)
    if max_candidates is not None:
        overrides["max_candidates"] = max_candidates
    if fractional_kelly is not None:
        overrides["fractional_kelly"] = fractional_kelly
    return StrategyConfig(**overrides)  # type: ignore[arg-type]
