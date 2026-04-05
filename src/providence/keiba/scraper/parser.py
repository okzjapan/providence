"""Generic fixed-length text parser for JRDB data files.

JRDB distributes data as CP932-encoded fixed-length text.
Each record occupies exactly `record_length` bytes (excluding CRLF).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    """Definition of one field within a fixed-length record.

    Attributes:
        name:   Column name in the output dict.
        start:  Start position in JRDB spec (1-based).
        length: Field width in bytes.
        dtype:  "str", "int", "float", or "hex".
        scale:  Multiplicative scale for float fields.
                Raw numeric value is multiplied by scale.
                Example: scale=0.1 means 555 → 55.5
    """

    name: str
    start: int
    length: int
    dtype: str = "str"
    scale: float = 1.0

    @property
    def offset(self) -> int:
        """0-based byte offset for Python slicing."""
        return self.start - 1

    @property
    def end(self) -> int:
        """0-based exclusive end position."""
        return self.offset + self.length


def parse_record(line: bytes, specs: list[FieldSpec]) -> dict[str, Any]:
    """Parse a single fixed-length record into a dict.

    Args:
        line:  One record as raw CP932 bytes (no trailing CRLF).
        specs: Field definitions to extract.

    Returns:
        dict mapping field name → parsed value (or None for missing data).
    """
    result: dict[str, Any] = {}
    for spec in specs:
        raw = line[spec.offset : spec.end]
        try:
            text = raw.decode("cp932", errors="replace").strip()
        except Exception:
            result[spec.name] = None
            continue
        result[spec.name] = _convert(text, spec.dtype, spec.scale)
    return result


def parse_file(
    data: bytes,
    specs: list[FieldSpec],
    record_length: int,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Parse an entire JRDB file into a list of record dicts.

    Lines are split on CRLF (or LF). Each line must be exactly
    `record_length` bytes after stripping the line ending.
    Lines that don't match are skipped (or raise if strict=True).

    Args:
        data:          Raw file content (CP932 bytes).
        specs:         Field definitions.
        record_length: Expected byte length per record (excluding CRLF).
        strict:        If True, raise on unexpected record lengths.
    """
    lines = data.replace(b"\r\n", b"\n").split(b"\n")
    records: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line:
            continue
        if len(line) != record_length:
            msg = f"Line {i}: expected {record_length} bytes, got {len(line)}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue
        records.append(parse_record(line, specs))
    return records


_HEX_MAP = {str(d): d for d in range(10)}
_HEX_MAP.update({"A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15})
_HEX_MAP.update({k.lower(): v for k, v in _HEX_MAP.items() if k.isalpha()})


def _convert(text: str, dtype: str, scale: float) -> Any:
    if not text or text == "-":
        return None

    if dtype == "str":
        return text or None

    if dtype == "int":
        try:
            return int(text)
        except ValueError:
            return None

    if dtype == "float":
        try:
            return float(text) * scale
        except ValueError:
            return None

    if dtype == "hex":
        return _HEX_MAP.get(text)

    raise ValueError(f"Unknown dtype: {dtype!r}")
