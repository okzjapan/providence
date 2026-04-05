"""JRDB KSA/KZA (騎手データ) and CSA/CZA (調教師データ) field definitions.

Both files have 272-byte records (excluding CRLF).
KSA = differential update, KZA = full dump.
CSA = differential update, CZA = full dump.
"""

from providence.keiba.scraper.parser import FieldSpec

KSA_RECORD_LENGTH = 270
CSA_RECORD_LENGTH = 270

KSA_FIELDS: list[FieldSpec] = [
    FieldSpec("jockey_code", start=1, length=5, dtype="str"),
    FieldSpec("retired_flag", start=6, length=1, dtype="int"),
    FieldSpec("retired_date", start=7, length=8, dtype="str"),
    FieldSpec("jockey_name", start=15, length=12, dtype="str"),
    FieldSpec("jockey_kana", start=27, length=30, dtype="str"),
    FieldSpec("jockey_short_name", start=57, length=6, dtype="str"),
    FieldSpec("affiliation_code", start=63, length=1, dtype="int"),  # 1=美浦,2=栗東
    FieldSpec("stable_name", start=64, length=4, dtype="str"),
    FieldSpec("birth_date", start=68, length=8, dtype="str"),
    FieldSpec("first_win_year", start=76, length=4, dtype="int"),
    FieldSpec("apprentice_class", start=80, length=1, dtype="int"),
    FieldSpec("stable_code", start=81, length=5, dtype="str"),
    # --- This year ---
    FieldSpec("this_year_leading", start=134, length=3, dtype="int"),
    FieldSpec("this_year_flat_1st", start=137, length=3, dtype="int"),
    FieldSpec("this_year_flat_2nd", start=140, length=3, dtype="int"),
    FieldSpec("this_year_flat_3rd", start=143, length=3, dtype="int"),
    FieldSpec("this_year_flat_unplaced", start=146, length=3, dtype="int"),
    # --- Last year ---
    FieldSpec("last_year_leading", start=167, length=3, dtype="int"),
    FieldSpec("last_year_flat_1st", start=170, length=3, dtype="int"),
    FieldSpec("last_year_flat_2nd", start=173, length=3, dtype="int"),
    FieldSpec("last_year_flat_3rd", start=176, length=3, dtype="int"),
    FieldSpec("last_year_flat_unplaced", start=179, length=3, dtype="int"),
    # --- Career totals ---
    FieldSpec("career_flat_1st", start=200, length=5, dtype="int"),
    FieldSpec("career_flat_2nd", start=205, length=5, dtype="int"),
    FieldSpec("career_flat_3rd", start=210, length=5, dtype="int"),
    FieldSpec("career_flat_unplaced", start=215, length=5, dtype="int"),
    # --- Data date ---
    FieldSpec("data_date", start=240, length=8, dtype="str"),
]

CSA_FIELDS: list[FieldSpec] = [
    FieldSpec("trainer_code", start=1, length=5, dtype="str"),
    FieldSpec("retired_flag", start=6, length=1, dtype="int"),
    FieldSpec("retired_date", start=7, length=8, dtype="str"),
    FieldSpec("trainer_name", start=15, length=12, dtype="str"),
    FieldSpec("trainer_kana", start=27, length=30, dtype="str"),
    FieldSpec("trainer_short_name", start=57, length=6, dtype="str"),
    FieldSpec("affiliation_code", start=63, length=1, dtype="int"),  # 1=美浦,2=栗東
    FieldSpec("stable_name", start=64, length=4, dtype="str"),
    FieldSpec("birth_date", start=68, length=8, dtype="str"),
    FieldSpec("first_win_year", start=76, length=4, dtype="int"),
    # --- This year ---
    FieldSpec("this_year_leading", start=128, length=3, dtype="int"),
    FieldSpec("this_year_flat_1st", start=131, length=3, dtype="int"),
    FieldSpec("this_year_flat_2nd", start=134, length=3, dtype="int"),
    FieldSpec("this_year_flat_3rd", start=137, length=3, dtype="int"),
    FieldSpec("this_year_flat_unplaced", start=140, length=3, dtype="int"),
    # --- Last year ---
    FieldSpec("last_year_leading", start=161, length=3, dtype="int"),
    FieldSpec("last_year_flat_1st", start=164, length=3, dtype="int"),
    FieldSpec("last_year_flat_2nd", start=167, length=3, dtype="int"),
    FieldSpec("last_year_flat_3rd", start=170, length=3, dtype="int"),
    FieldSpec("last_year_flat_unplaced", start=173, length=3, dtype="int"),
    # --- Career totals ---
    FieldSpec("career_flat_1st", start=194, length=5, dtype="int"),
    FieldSpec("career_flat_2nd", start=199, length=5, dtype="int"),
    FieldSpec("career_flat_3rd", start=204, length=5, dtype="int"),
    FieldSpec("career_flat_unplaced", start=209, length=5, dtype="int"),
    # --- Data date ---
    FieldSpec("data_date", start=234, length=8, dtype="str"),
]
