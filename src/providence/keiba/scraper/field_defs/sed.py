"""JRDB SED (成績データ) field definitions.

Record length: 374 bytes (excluding CRLF). Spec says 376 but includes CRLF.
One record per horse per race. Updated Thursday 16:00.
Spec: https://jrdb.com/program/Sed/sed_doc.txt (第4版a 2022.08.22)
"""

from providence.keiba.scraper.parser import FieldSpec

RECORD_LENGTH = 374

FIELDS: list[FieldSpec] = [
    # --- Race key (8 bytes) ---
    FieldSpec("place_code", start=1, length=2, dtype="str"),
    FieldSpec("year", start=3, length=2, dtype="str"),
    FieldSpec("kai", start=5, length=1, dtype="str"),
    FieldSpec("day", start=6, length=1, dtype="hex"),
    FieldSpec("race_number", start=7, length=2, dtype="str"),
    # --- Horse identity ---
    FieldSpec("post_position", start=9, length=2, dtype="int"),
    FieldSpec("blood_registration_number", start=11, length=8, dtype="str"),
    FieldSpec("race_date", start=19, length=8, dtype="str"),  # YYYYMMDD
    FieldSpec("horse_name", start=27, length=36, dtype="str"),
    # --- Race conditions ---
    FieldSpec("distance", start=63, length=4, dtype="int"),
    FieldSpec("surface_code", start=67, length=1, dtype="int"),  # 1=芝,2=ダート,3=障害
    FieldSpec("direction_code", start=68, length=1, dtype="int"),
    FieldSpec("course_type_code", start=69, length=1, dtype="int"),
    FieldSpec("going_code", start=70, length=2, dtype="int"),
    FieldSpec("grade_code", start=80, length=1, dtype="int"),
    FieldSpec("race_name", start=81, length=50, dtype="str"),
    FieldSpec("num_runners", start=131, length=2, dtype="int"),
    # --- Result ---
    FieldSpec("finish_position", start=141, length=2, dtype="int"),
    FieldSpec("abnormality_code", start=143, length=1, dtype="int"),
    FieldSpec("race_time", start=144, length=4, dtype="int"),  # 1byte:min, 2-4byte:sec*10
    FieldSpec("impost_weight", start=148, length=3, dtype="float", scale=0.1),
    # --- Odds ---
    FieldSpec("confirmed_win_odds", start=175, length=6, dtype="float"),
    FieldSpec("confirmed_popularity", start=181, length=2, dtype="int"),
    # --- JRDB analysis ---
    FieldSpec("jrdb_idm", start=183, length=3, dtype="int"),
    FieldSpec("ten_index", start=224, length=5, dtype="float"),
    FieldSpec("agari_index", start=229, length=5, dtype="float"),
    FieldSpec("pace_index", start=234, length=5, dtype="float"),
    FieldSpec("winner_or_second_margin", start=256, length=3, dtype="float", scale=0.1),
    FieldSpec("first_3f_time", start=259, length=3, dtype="float", scale=0.1),
    FieldSpec("last_3f_time", start=262, length=3, dtype="float", scale=0.1),
    # --- 2nd edition additions ---
    FieldSpec("corner_1_pos", start=309, length=2, dtype="int"),
    FieldSpec("corner_2_pos", start=311, length=2, dtype="int"),
    FieldSpec("corner_3_pos", start=313, length=2, dtype="int"),
    FieldSpec("corner_4_pos", start=315, length=2, dtype="int"),
    FieldSpec("jockey_code", start=323, length=5, dtype="str"),
    FieldSpec("trainer_code", start=328, length=5, dtype="str"),
    # --- 3rd edition additions ---
    FieldSpec("body_weight", start=333, length=3, dtype="int"),
    FieldSpec("body_weight_change", start=336, length=3, dtype="str"),  # sign+digits
    FieldSpec("weather_code", start=339, length=1, dtype="int"),
    FieldSpec("running_style_code", start=341, length=1, dtype="str"),
]
