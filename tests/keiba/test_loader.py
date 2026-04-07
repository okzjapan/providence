"""Tests for KeibaDataLoader."""

from datetime import date
from pathlib import Path

import pytest

_DB_PATH = Path("data/providence.db")
_HAS_DB = _DB_PATH.exists()

pytestmark = pytest.mark.skipif(not _HAS_DB, reason="Keiba DB not available")


@pytest.fixture()
def loader():
    from sqlalchemy import create_engine

    from providence.keiba.features.loader import KeibaDataLoader

    engine = create_engine(f"sqlite:///{_DB_PATH}")
    return KeibaDataLoader(engine=engine)


class TestLoadRaceDataset:
    def test_returns_dataframe_with_expected_columns(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )
        assert not df.is_empty()
        required = [
            "race_id", "race_key", "race_date", "racecourse_id", "distance",
            "surface_code", "going_code", "class_code", "field_size",
            "entry_id", "post_position", "blood_registration_number",
            "jockey_code", "trainer_code", "impost_weight", "idm",
            "finish_position", "race_time_sec", "last_3f_time",
            "corner_1_pos", "corner_4_pos",
            "body_weight", "body_weight_change",
            "horse_age", "sex_code", "sire_code",
            "row_key",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_race_time_sec_conversion(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )
        valid = df.filter(df["race_time_sec"].is_not_null())
        assert not valid.is_empty()
        times = valid["race_time_sec"].to_list()
        for t in times[:20]:
            assert 50.0 < t < 300.0, f"race_time_sec out of range: {t}"

    def test_horse_age_in_valid_range(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 7),
        )
        valid = df.filter(df["horse_age"].is_not_null())
        ages = valid["horse_age"].to_list()
        for a in ages[:50]:
            assert 2 <= a <= 12, f"horse_age out of range: {a}"

    def test_body_weight_change_parsed_as_int(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )
        import polars as pl

        assert df["body_weight_change"].dtype == pl.Int32

    def test_excludes_obstacle_races(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        surface_codes = df["surface_code"].unique().to_list()
        assert 3 not in surface_codes, "Obstacle races should be excluded"

    def test_date_range_filter(self, loader):
        df = loader.load_race_dataset(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
        )
        dates = df["race_date"].unique().sort()
        assert dates[0] >= date(2024, 6, 1)
        assert dates[-1] <= date(2024, 6, 30)
