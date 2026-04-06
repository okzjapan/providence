"""Tests for keiba relation features (jockey/trainer/sire)."""

from datetime import date

import polars as pl

from providence.keiba.features.relations import add_relation_features


def _make_df(rows):
    return pl.DataFrame(rows).with_columns(pl.col("race_date").cast(pl.Date))


class TestAddRelationFeatures:
    def test_empty(self):
        result = add_relation_features(pl.DataFrame())
        assert result.is_empty()

    def test_jockey_win_rate(self):
        rows = [
            {"jockey_code": "J001", "trainer_code": "T001", "sire_code": "S01",
             "race_date": date(2024, 1, i), "race_number": 1, "post_position": 1,
             "finish_position": 1 if i <= 3 else 5,
             "racecourse_id": 5, "surface_code": 1}
            for i in range(1, 11)
        ]
        df = _make_df(rows)
        result = add_relation_features(df)
        last_jwr = result["jockey_win_rate"][-1]
        assert last_jwr is not None
        assert 0.0 <= last_jwr <= 1.0

    def test_no_future_leakage(self):
        rows = [
            {"jockey_code": "J001", "trainer_code": "T001", "sire_code": "S01",
             "race_date": date(2024, 1, 1), "race_number": 1, "post_position": 1,
             "finish_position": 1, "racecourse_id": 5, "surface_code": 1},
        ]
        df = _make_df(rows)
        result = add_relation_features(df)
        assert result["jockey_win_rate"][0] is None
        assert result["trainer_win_rate"][0] is None
