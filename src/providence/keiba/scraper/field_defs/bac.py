"""JRDB BAC (番組データ) field definitions.

Record length: 176 bytes (excluding CRLF).
One record per race.
"""

from providence.keiba.scraper.parser import FieldSpec

RECORD_LENGTH = 182

FIELDS: list[FieldSpec] = [
    # --- Race key (8 bytes) ---
    FieldSpec("place_code", start=1, length=2, dtype="str"),
    FieldSpec("year", start=3, length=2, dtype="str"),
    FieldSpec("kai", start=5, length=1, dtype="str"),
    FieldSpec("day", start=6, length=1, dtype="hex"),
    FieldSpec("race_number", start=7, length=2, dtype="str"),
    # --- Race info ---
    FieldSpec("race_date", start=9, length=8, dtype="str"),       # YYYYMMDD
    FieldSpec("post_time", start=17, length=4, dtype="str"),      # HHMM
    FieldSpec("distance", start=21, length=4, dtype="int"),       # meters
    FieldSpec("surface_code", start=25, length=1, dtype="int"),   # 1=芝,2=ダート,3=障害
    FieldSpec("direction_code", start=26, length=1, dtype="int"), # 1=右,2=左,3=直
    FieldSpec("course_type_code", start=27, length=1, dtype="int"),  # 1=内,2=外,etc
    FieldSpec("age_restriction", start=28, length=2, dtype="str"),
    FieldSpec("class_code", start=30, length=2, dtype="str"),
    FieldSpec("weight_rule_code", start=32, length=1, dtype="str"),
    FieldSpec("weather_code", start=35, length=1, dtype="int"),
    FieldSpec("going_code", start=36, length=1, dtype="int"),     # 馬場状態
    FieldSpec("race_name", start=37, length=50, dtype="str"),
    FieldSpec("num_runners", start=95, length=2, dtype="int"),
]
