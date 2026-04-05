"""JRDB HJC (払戻情報データ) field definitions.

Record length: 444 bytes (excluding CRLF).
One record per race. Updated Saturday/Sunday 17:00.

Note: Since 2024.09.29, trifecta payout fields changed from 8 to 9 digits.
The current definitions reflect the post-2024 format.
For pre-2024 data, use HJC_FIELDS_LEGACY.
"""

from providence.keiba.scraper.parser import FieldSpec

RECORD_LENGTH = 442
RECORD_LENGTH_LEGACY = 438  # pre-2024 format

FIELDS: list[FieldSpec] = [
    # --- Race key (8 bytes) ---
    FieldSpec("place_code", start=1, length=2, dtype="str"),
    FieldSpec("year", start=3, length=2, dtype="str"),
    FieldSpec("kai", start=5, length=1, dtype="str"),
    FieldSpec("day", start=6, length=1, dtype="hex"),
    FieldSpec("race_number", start=7, length=2, dtype="str"),
    # --- Win (単勝): 3 horses x 9 bytes (horse_number 2 + payout 7) ---
    FieldSpec("win_horse_1", start=9, length=2, dtype="int"),
    FieldSpec("win_payout_1", start=11, length=7, dtype="int"),
    FieldSpec("win_horse_2", start=18, length=2, dtype="int"),
    FieldSpec("win_payout_2", start=20, length=7, dtype="int"),
    FieldSpec("win_horse_3", start=27, length=2, dtype="int"),
    FieldSpec("win_payout_3", start=29, length=7, dtype="int"),
    # --- Place (複勝): 5 horses x 9 bytes ---
    FieldSpec("place_horse_1", start=36, length=2, dtype="int"),
    FieldSpec("place_payout_1", start=38, length=7, dtype="int"),
    FieldSpec("place_horse_2", start=45, length=2, dtype="int"),
    FieldSpec("place_payout_2", start=47, length=7, dtype="int"),
    FieldSpec("place_horse_3", start=54, length=2, dtype="int"),
    FieldSpec("place_payout_3", start=56, length=7, dtype="int"),
    FieldSpec("place_horse_4", start=63, length=2, dtype="int"),
    FieldSpec("place_payout_4", start=65, length=7, dtype="int"),
    FieldSpec("place_horse_5", start=72, length=2, dtype="int"),
    FieldSpec("place_payout_5", start=74, length=7, dtype="int"),
    # --- Bracket Quinella (枠連): 3 combos x 9 bytes ---
    FieldSpec("bracket_quinella_combo_1", start=81, length=4, dtype="str"),
    FieldSpec("bracket_quinella_payout_1", start=85, length=7, dtype="int"),
    # --- Quinella (馬連): 3 combos x 12 bytes (combo 4 + payout 8) ---
    FieldSpec("quinella_combo_1", start=108, length=4, dtype="str"),
    FieldSpec("quinella_payout_1", start=112, length=8, dtype="int"),
    FieldSpec("quinella_combo_2", start=120, length=4, dtype="str"),
    FieldSpec("quinella_payout_2", start=124, length=8, dtype="int"),
    FieldSpec("quinella_combo_3", start=132, length=4, dtype="str"),
    FieldSpec("quinella_payout_3", start=136, length=8, dtype="int"),
    # --- Wide (ワイド): 7 combos x 12 bytes ---
    FieldSpec("wide_combo_1", start=144, length=4, dtype="str"),
    FieldSpec("wide_payout_1", start=148, length=8, dtype="int"),
    FieldSpec("wide_combo_2", start=156, length=4, dtype="str"),
    FieldSpec("wide_payout_2", start=160, length=8, dtype="int"),
    FieldSpec("wide_combo_3", start=168, length=4, dtype="str"),
    FieldSpec("wide_payout_3", start=172, length=8, dtype="int"),
    # --- Exacta (馬単): 6 combos x 12 bytes ---
    FieldSpec("exacta_combo_1", start=228, length=4, dtype="str"),
    FieldSpec("exacta_payout_1", start=232, length=8, dtype="int"),
    FieldSpec("exacta_combo_2", start=240, length=4, dtype="str"),
    FieldSpec("exacta_payout_2", start=244, length=8, dtype="int"),
    FieldSpec("exacta_combo_3", start=252, length=4, dtype="str"),
    FieldSpec("exacta_payout_3", start=256, length=8, dtype="int"),
    # --- Trio (3連複): 3 combos x 14 bytes (combo 6 + payout 8) ---
    FieldSpec("trio_combo_1", start=300, length=6, dtype="str"),
    FieldSpec("trio_payout_1", start=306, length=8, dtype="int"),
    FieldSpec("trio_combo_2", start=314, length=6, dtype="str"),
    FieldSpec("trio_payout_2", start=320, length=8, dtype="int"),
    FieldSpec("trio_combo_3", start=328, length=6, dtype="str"),
    FieldSpec("trio_payout_3", start=334, length=8, dtype="int"),
    # --- Trifecta (3連単): 6 combos x 15 bytes (combo 6 + payout 9) ---
    FieldSpec("trifecta_combo_1", start=342, length=6, dtype="str"),
    FieldSpec("trifecta_payout_1", start=348, length=9, dtype="int"),
    FieldSpec("trifecta_combo_2", start=357, length=6, dtype="str"),
    FieldSpec("trifecta_payout_2", start=363, length=9, dtype="int"),
    FieldSpec("trifecta_combo_3", start=372, length=6, dtype="str"),
    FieldSpec("trifecta_payout_3", start=378, length=9, dtype="int"),
    FieldSpec("trifecta_combo_4", start=387, length=6, dtype="str"),
    FieldSpec("trifecta_payout_4", start=393, length=9, dtype="int"),
    FieldSpec("trifecta_combo_5", start=402, length=6, dtype="str"),
    FieldSpec("trifecta_payout_5", start=408, length=9, dtype="int"),
    FieldSpec("trifecta_combo_6", start=417, length=6, dtype="str"),
    FieldSpec("trifecta_payout_6", start=423, length=9, dtype="int"),
]
