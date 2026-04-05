"""JRDB KYI (競走馬データ) field definitions.

Record length: 1022 bytes (excluding CRLF). Spec says 1024 but includes CRLF.
One record per horse per race. Updated Friday/Saturday 19:00.
Spec: https://jrdb.com/program/Kyi/kyi_doc.txt (第11版 2023.06.25)
"""

from providence.keiba.scraper.parser import FieldSpec

RECORD_LENGTH = 1022

FIELDS: list[FieldSpec] = [
    # --- Race key (8 bytes, start=1) ---
    FieldSpec("place_code", start=1, length=2, dtype="str"),
    FieldSpec("year", start=3, length=2, dtype="str"),
    FieldSpec("kai", start=5, length=1, dtype="str"),
    FieldSpec("day", start=6, length=1, dtype="hex"),
    FieldSpec("race_number", start=7, length=2, dtype="str"),
    # --- Horse identity ---
    FieldSpec("post_position", start=9, length=2, dtype="int"),
    FieldSpec("blood_registration_number", start=11, length=8, dtype="str"),
    FieldSpec("horse_name", start=19, length=36, dtype="str"),
    # --- JRDB indices (ZZ9.9 format = already decimal, scale=1.0) ---
    FieldSpec("idm", start=55, length=5, dtype="float"),
    FieldSpec("jockey_index", start=60, length=5, dtype="float"),
    FieldSpec("info_index", start=65, length=5, dtype="float"),
    FieldSpec("composite_index", start=85, length=5, dtype="float"),
    # --- Running style / aptitude ---
    FieldSpec("running_style_code", start=90, length=1, dtype="int"),
    FieldSpec("distance_aptitude_code", start=91, length=1, dtype="str"),
    # --- Odds ---
    FieldSpec("base_win_odds", start=96, length=5, dtype="float"),
    FieldSpec("base_place_odds", start=103, length=5, dtype="float"),
    # --- More indices ---
    FieldSpec("popularity_index", start=140, length=5, dtype="int"),
    FieldSpec("training_index", start=145, length=5, dtype="float"),
    FieldSpec("stable_index", start=150, length=5, dtype="float"),
    # --- Equipment ---
    FieldSpec("blinkers", start=171, length=1, dtype="str"),
    # --- Weight / jockey / trainer ---
    FieldSpec("impost_weight", start=184, length=3, dtype="float", scale=0.1),
    FieldSpec("apprentice_class", start=187, length=1, dtype="int"),
    # --- Codes (5th edition, start=336) ---
    FieldSpec("turf_aptitude_code", start=334, length=1, dtype="str"),
    FieldSpec("dirt_aptitude_code", start=335, length=1, dtype="str"),
    FieldSpec("jockey_code", start=336, length=5, dtype="str"),
    FieldSpec("trainer_code", start=341, length=5, dtype="str"),
    # --- Class / conditions (6th edition) ---
    FieldSpec("class_code", start=358, length=1, dtype="str"),
    # --- Cancel flag (7th edition) ---
    FieldSpec("cancel_flag", start=403, length=1, dtype="int"),
]
