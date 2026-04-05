"""Tests for JRDB race key utilities."""

import pytest

from providence.keiba.race_key import build_race_key, parse_race_key


class TestParseRaceKey:
    def test_basic(self):
        result = parse_race_key("06050211")
        assert result == {
            "place_code": "06",
            "year": "05",
            "kai": "0",
            "day": 2,
            "race_number": "11",
        }

    def test_hex_day_a(self):
        result = parse_race_key("06050A11")
        assert result["kai"] == "0"
        assert result["day"] == 10

    def test_hex_day_c(self):
        result = parse_race_key("06050C11")
        assert result["day"] == 12

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="8 characters"):
            parse_race_key("123")


class TestBuildRaceKey:
    def test_basic(self):
        key = build_race_key("06", "05", "0", 2, "11")
        assert key == "06050211"

    def test_hex_day(self):
        key = build_race_key("06", "05", "0", 10, "11")
        assert key == "06050A11"

    def test_roundtrip(self):
        original = "06050A11"
        parsed = parse_race_key(original)
        rebuilt = build_race_key(
            parsed["place_code"],
            parsed["year"],
            parsed["kai"],
            parsed["day"],
            parsed["race_number"],
        )
        assert rebuilt == original

    def test_invalid_day(self):
        with pytest.raises(ValueError, match="day must be 1-12"):
            build_race_key("06", "05", "0", 0, "11")
