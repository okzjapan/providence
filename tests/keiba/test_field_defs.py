"""Tests for JRDB field definitions.

Validates that field specs are internally consistent:
- No field exceeds the declared record length
- No duplicate field names within a file
"""

import pytest

from providence.keiba.scraper.field_defs.bac import (
    FIELDS as BAC_FIELDS,
)
from providence.keiba.scraper.field_defs.bac import (
    RECORD_LENGTH as BAC_LEN,
)
from providence.keiba.scraper.field_defs.hjc import (
    FIELDS as HJC_FIELDS,
)
from providence.keiba.scraper.field_defs.hjc import (
    RECORD_LENGTH as HJC_LEN,
)
from providence.keiba.scraper.field_defs.kyi import (
    FIELDS as KYI_FIELDS,
)
from providence.keiba.scraper.field_defs.kyi import (
    RECORD_LENGTH as KYI_LEN,
)
from providence.keiba.scraper.field_defs.master import (
    CSA_FIELDS,
    CSA_RECORD_LENGTH,
    KSA_FIELDS,
    KSA_RECORD_LENGTH,
)
from providence.keiba.scraper.field_defs.sed import (
    FIELDS as SED_FIELDS,
)
from providence.keiba.scraper.field_defs.sed import (
    RECORD_LENGTH as SED_LEN,
)
from providence.keiba.scraper.field_defs.ukc import (
    FIELDS as UKC_FIELDS,
)
from providence.keiba.scraper.field_defs.ukc import (
    RECORD_LENGTH as UKC_LEN,
)

ALL_DEFS = [
    ("BAC", BAC_FIELDS, BAC_LEN),
    ("KYI", KYI_FIELDS, KYI_LEN),
    ("SED", SED_FIELDS, SED_LEN),
    ("HJC", HJC_FIELDS, HJC_LEN),
    ("UKC", UKC_FIELDS, UKC_LEN),
    ("KSA", KSA_FIELDS, KSA_RECORD_LENGTH),
    ("CSA", CSA_FIELDS, CSA_RECORD_LENGTH),
]


@pytest.mark.parametrize("name,fields,record_length", ALL_DEFS, ids=[d[0] for d in ALL_DEFS])
class TestFieldDefs:
    def test_no_field_exceeds_record_length(self, name, fields, record_length):
        for spec in fields:
            end = spec.offset + spec.length
            assert end <= record_length, (
                f"{name}.{spec.name}: end byte {end} exceeds record length {record_length}"
            )

    def test_no_duplicate_names(self, name, fields, record_length):
        names = [f.name for f in fields]
        dupes = [n for n in names if names.count(n) > 1]
        assert not dupes, f"{name} has duplicate field names: {set(dupes)}"

    def test_all_starts_positive(self, name, fields, record_length):
        for spec in fields:
            assert spec.start >= 1, f"{name}.{spec.name}: start must be >= 1, got {spec.start}"

    def test_all_lengths_positive(self, name, fields, record_length):
        for spec in fields:
            assert spec.length >= 1, f"{name}.{spec.name}: length must be >= 1, got {spec.length}"

    def test_valid_dtypes(self, name, fields, record_length):
        valid = {"str", "int", "float", "hex"}
        for spec in fields:
            assert spec.dtype in valid, f"{name}.{spec.name}: invalid dtype {spec.dtype!r}"
