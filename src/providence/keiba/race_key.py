"""JRDB race key parsing and construction utilities.

JRDB race key format: 8 bytes total
  - place_code: 2 bytes (JRA racecourse code, e.g. '06' = Hanshin)
  - year:       2 bytes (last 2 digits, e.g. '05' = 2005)
  - kai:        1 byte  (開催回, '1'-'9')
  - day:        1 byte  (開催日, hex: '1'-'9','A'=10,'B'=11,'C'=12)
  - race_number:2 bytes (R番号, '01'-'12')
"""

from __future__ import annotations

_HEX_TO_INT = {str(i): i for i in range(0, 10)}
_HEX_TO_INT.update({"A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15})
_HEX_TO_INT.update({k.lower(): v for k, v in _HEX_TO_INT.items() if k.isalpha()})

_INT_TO_HEX = {v: k.upper() for k, v in _HEX_TO_INT.items() if k.isupper() or k.isdigit()}


def parse_race_key(key: str) -> dict[str, str | int]:
    """Parse an 8-character JRDB race key into components.

    Returns dict with: place_code (str), year (str), kai (str),
    day (int), race_number (str).
    """
    if len(key) != 8:
        raise ValueError(f"Race key must be 8 characters, got {len(key)}: {key!r}")
    day_char = key[5]
    day_val = _HEX_TO_INT.get(day_char)
    if day_val is None:
        raise ValueError(f"Invalid day character in race key: {day_char!r}")
    return {
        "place_code": key[0:2],
        "year": key[2:4],
        "kai": key[4],
        "day": day_val,
        "race_number": key[6:8],
    }


def build_race_key(place_code: str, year: str, kai: str, day: int, race_number: str) -> str:
    """Build an 8-character JRDB race key from components.

    day is an integer (1-12) converted to hex single digit.
    """
    if not (1 <= day <= 12):
        raise ValueError(f"day must be 1-12, got {day}")
    day_hex = _INT_TO_HEX[day]
    return f"{place_code}{year}{kai}{day_hex}{race_number}"
