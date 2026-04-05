"""Tests for the JRDB fixed-length parser."""

import pytest

from providence.keiba.scraper.parser import FieldSpec, parse_file, parse_record


class TestFieldSpec:
    def test_offset_and_end(self):
        spec = FieldSpec(name="f", start=3, length=5)
        assert spec.offset == 2
        assert spec.end == 7

    def test_start_1_based(self):
        spec = FieldSpec(name="f", start=1, length=2)
        assert spec.offset == 0
        assert spec.end == 2


class TestParseRecordStr:
    def test_basic_string(self):
        specs = [FieldSpec("name", start=1, length=5, dtype="str")]
        line = "Hello".encode("cp932")
        result = parse_record(line, specs)
        assert result["name"] == "Hello"

    def test_string_strips_whitespace(self):
        specs = [FieldSpec("name", start=1, length=8, dtype="str")]
        line = "  abc   ".encode("cp932")
        result = parse_record(line, specs)
        assert result["name"] == "abc"

    def test_empty_string_is_none(self):
        specs = [FieldSpec("name", start=1, length=4, dtype="str")]
        line = b"    "
        result = parse_record(line, specs)
        assert result["name"] is None

    def test_japanese_cp932(self):
        specs = [FieldSpec("name", start=1, length=8, dtype="str")]
        text = "テスト"
        line = text.encode("cp932")
        padded = line + b" " * (8 - len(line))
        result = parse_record(padded, specs)
        assert result["name"] == "テスト"


class TestParseRecordInt:
    def test_basic_int(self):
        specs = [FieldSpec("num", start=1, length=3, dtype="int")]
        line = b"042"
        result = parse_record(line, specs)
        assert result["num"] == 42

    def test_int_with_spaces_is_none(self):
        specs = [FieldSpec("num", start=1, length=3, dtype="int")]
        line = b"   "
        result = parse_record(line, specs)
        assert result["num"] is None

    def test_invalid_int_is_none(self):
        specs = [FieldSpec("num", start=1, length=3, dtype="int")]
        line = b"abc"
        result = parse_record(line, specs)
        assert result["num"] is None

    def test_dash_is_none(self):
        specs = [FieldSpec("num", start=1, length=1, dtype="int")]
        line = b"-"
        result = parse_record(line, specs)
        assert result["num"] is None


class TestParseRecordFloat:
    def test_basic_float(self):
        specs = [FieldSpec("val", start=1, length=5, dtype="float")]
        line = b"123.4"
        result = parse_record(line, specs)
        assert result["val"] == pytest.approx(123.4)

    def test_float_with_scale(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="float", scale=0.1)]
        line = b"555"
        result = parse_record(line, specs)
        assert result["val"] == pytest.approx(55.5)

    def test_float_spaces_is_none(self):
        specs = [FieldSpec("val", start=1, length=5, dtype="float")]
        line = b"     "
        result = parse_record(line, specs)
        assert result["val"] is None


class TestParseRecordHex:
    def test_digits_1_to_9(self):
        specs = [FieldSpec("day", start=1, length=1, dtype="hex")]
        for digit in range(1, 10):
            line = str(digit).encode("ascii")
            result = parse_record(line, specs)
            assert result["day"] == digit

    def test_hex_a_to_c(self):
        specs = [FieldSpec("day", start=1, length=1, dtype="hex")]
        for char, expected in [("A", 10), ("B", 11), ("C", 12)]:
            assert parse_record(char.encode(), specs)["day"] == expected

    def test_hex_lowercase(self):
        specs = [FieldSpec("day", start=1, length=1, dtype="hex")]
        assert parse_record(b"a", specs)["day"] == 10


class TestParseRecordMultipleFields:
    def test_multiple_fields(self):
        specs = [
            FieldSpec("code", start=1, length=2, dtype="str"),
            FieldSpec("year", start=3, length=2, dtype="int"),
            FieldSpec("day", start=5, length=1, dtype="hex"),
        ]
        line = b"0605A"
        result = parse_record(line, specs)
        assert result == {"code": "06", "year": 5, "day": 10}


class TestParseFile:
    def _make_file(self, *lines: bytes) -> bytes:
        return b"\r\n".join(lines)

    def test_basic_file(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="int")]
        data = self._make_file(b"001", b"002", b"003")
        records = parse_file(data, specs, record_length=3)
        assert len(records) == 3
        assert records[0]["val"] == 1
        assert records[2]["val"] == 3

    def test_skips_wrong_length_in_non_strict(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="int")]
        data = self._make_file(b"001", b"bad_line_too_long", b"003")
        records = parse_file(data, specs, record_length=3)
        assert len(records) == 2

    def test_raises_on_wrong_length_in_strict(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="int")]
        data = self._make_file(b"001", b"bad_line_too_long")
        with pytest.raises(ValueError, match="expected 3 bytes"):
            parse_file(data, specs, record_length=3, strict=True)

    def test_empty_trailing_line(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="int")]
        data = b"001\r\n002\r\n"
        records = parse_file(data, specs, record_length=3)
        assert len(records) == 2

    def test_lf_only(self):
        specs = [FieldSpec("val", start=1, length=3, dtype="int")]
        data = b"001\n002\n"
        records = parse_file(data, specs, record_length=3)
        assert len(records) == 2
