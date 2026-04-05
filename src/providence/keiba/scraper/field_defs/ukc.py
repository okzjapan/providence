"""JRDB UKC (馬基本データ) field definitions.

Record length: 290 bytes (excluding CRLF).
One record per horse. Contains bloodline information.
"""

from providence.keiba.scraper.parser import FieldSpec

RECORD_LENGTH = 290

FIELDS: list[FieldSpec] = [
    FieldSpec("blood_registration_number", start=1, length=8, dtype="str"),
    FieldSpec("horse_name", start=9, length=36, dtype="str"),
    FieldSpec("sex_code", start=45, length=1, dtype="int"),     # 1=牡,2=牝,3=セン
    FieldSpec("coat_color_code", start=46, length=2, dtype="int"),
    FieldSpec("horse_symbol_code", start=48, length=2, dtype="int"),
    FieldSpec("sire_name", start=50, length=36, dtype="str"),
    FieldSpec("dam_name", start=86, length=36, dtype="str"),
    FieldSpec("broodmare_sire_name", start=122, length=36, dtype="str"),
    FieldSpec("birth_date", start=158, length=8, dtype="str"),  # YYYYMMDD
    FieldSpec("sire_birth_year", start=166, length=4, dtype="int"),
    FieldSpec("dam_birth_year", start=170, length=4, dtype="int"),
    FieldSpec("broodmare_sire_birth_year", start=174, length=4, dtype="int"),
    FieldSpec("owner_name", start=178, length=40, dtype="str"),
    FieldSpec("owner_code", start=218, length=2, dtype="str"),
    FieldSpec("breeder_name", start=220, length=40, dtype="str"),
    FieldSpec("birthplace", start=260, length=8, dtype="str"),
    FieldSpec("retired_flag", start=268, length=1, dtype="int"),  # 0=現役,1=抹消
    FieldSpec("data_date", start=269, length=8, dtype="str"),
    FieldSpec("sire_code", start=277, length=4, dtype="str"),
    FieldSpec("broodmare_sire_code", start=281, length=4, dtype="str"),
]
