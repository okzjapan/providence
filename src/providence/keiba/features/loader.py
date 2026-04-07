"""Data loading utilities for keiba feature engineering."""

from __future__ import annotations

from datetime import date

import polars as pl

from providence.database.engine import get_engine


class KeibaDataLoader:
    """Load JRA horse racing data from DB into Polars DataFrames."""

    def __init__(self, engine=None) -> None:
        self.engine = engine or get_engine()

    def load_race_dataset(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Return one row per race entry with joined race/result/horse/jockey/trainer data.

        Excludes obstacle races (surface_code=3).
        end_date is inclusive.
        """
        where_clauses = ["r.surface_code IN (1, 2)"]
        if start_date is not None:
            where_clauses.append(f"r.race_date >= '{start_date.isoformat()}'")
        if end_date is not None:
            where_clauses.append(f"r.race_date <= '{end_date.isoformat()}'")

        where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT
            r.id AS race_id,
            r.race_key,
            r.race_date,
            r.racecourse_id,
            r.race_number,
            r.distance,
            r.surface_code,
            r.direction_code,
            r.course_type_code,
            r.going_code,
            r.weather_code,
            r.class_code,
            r.weight_rule_code,
            r.num_runners AS field_size,
            re.id AS entry_id,
            re.post_position,
            re.blood_registration_number,
            re.jockey_code,
            re.trainer_code,
            re.impost_weight,
            re.base_win_odds,
            re.idm,
            re.jockey_index,
            re.info_index,
            re.training_index,
            re.stable_index,
            re.composite_index,
            re.running_style_code,
            re.distance_aptitude_code,
            rr.finish_position,
            CASE WHEN rr.race_time IS NOT NULL
                 THEN (CAST(rr.race_time AS INTEGER) / 1000) * 60
                      + (CAST(rr.race_time AS INTEGER) % 1000) / 10.0
                 ELSE NULL END AS race_time_sec,
            rr.race_time AS race_time_raw,
            rr.last_3f_time,
            rr.first_3f_time,
            rr.margin,
            rr.corner_1_pos,
            rr.corner_2_pos,
            rr.corner_3_pos,
            rr.corner_4_pos,
            rr.confirmed_win_odds,
            rr.confirmed_popularity,
            rr.body_weight,
            rr.body_weight_change AS body_weight_change_raw,
            h.sex_code,
            h.birth_date AS horse_birth_date,
            h.sire_code,
            h.broodmare_sire_code,
            CASE WHEN h.birth_date IS NOT NULL
                 THEN CAST(strftime('%Y', r.race_date) AS INTEGER)
                      - CAST(SUBSTR(h.birth_date, 1, 4) AS INTEGER)
                 ELSE NULL END AS horse_age,
            j.affiliation_code AS jockey_affiliation,
            j.apprentice_class,
            t.affiliation_code AS trainer_affiliation
        FROM keiba_race_entries re
        JOIN keiba_races r ON re.race_id = r.id
        LEFT JOIN keiba_race_results rr ON rr.entry_id = re.id
        LEFT JOIN keiba_horses h ON re.horse_id = h.id
        LEFT JOIN keiba_jockeys j ON re.jockey_id = j.id
        LEFT JOIN keiba_trainers t ON re.trainer_id = t.id
        {where_sql}
        ORDER BY r.race_date, r.race_number, re.post_position
        """

        schema_overrides = {
            "class_code": pl.Utf8,
            "weight_rule_code": pl.Utf8,
            "distance_aptitude_code": pl.Utf8,
            "blood_registration_number": pl.Utf8,
            "jockey_code": pl.Utf8,
            "trainer_code": pl.Utf8,
            "race_key": pl.Utf8,
            "horse_birth_date": pl.Utf8,
            "sire_code": pl.Utf8,
            "broodmare_sire_code": pl.Utf8,
            "body_weight_change_raw": pl.Utf8,
        }

        with self.engine.connect() as conn:
            df = pl.read_database(
                query=query,
                connection=conn,
                schema_overrides=schema_overrides,
                infer_schema_length=1000,
            )

        if df.is_empty():
            return df

        if df["race_date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("race_date").str.strptime(pl.Date, strict=False))
        elif df["race_date"].dtype != pl.Date:
            df = df.with_columns(pl.col("race_date").cast(pl.Date))

        df = df.with_columns(
            pl.col("body_weight_change_raw")
            .cast(pl.Utf8)
            .str.replace_all(" ", "")
            .cast(pl.Int32, strict=False)
            .alias("body_weight_change"),
        )

        df = df.with_columns(
            pl.concat_str([
                pl.col("race_date").cast(pl.Utf8),
                pl.lit("-"),
                pl.col("race_number").cast(pl.Utf8),
                pl.lit("-"),
                pl.col("post_position").cast(pl.Utf8),
            ]).alias("row_key"),
        )

        return df

    def load_all(self) -> pl.DataFrame:
        return self.load_race_dataset()
