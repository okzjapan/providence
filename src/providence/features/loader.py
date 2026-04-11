"""Data loading utilities for feature engineering."""

from __future__ import annotations

from datetime import date

import polars as pl

from providence.database.engine import get_engine


class DataLoader:
    """Load race data from SQLite into polars DataFrames."""

    def __init__(self, engine=None) -> None:
        self.engine = engine or get_engine()

    def load_race_dataset(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Return one row per race entry with joined race/result/rider data.

        `end_date` is inclusive for convenience. In callers that need
        strict temporal cutoff, pass the prior day.
        """

        where_clauses: list[str] = []
        if start_date is not None:
            where_clauses.append(f"r.race_date >= '{start_date.isoformat()}'")
        if end_date is not None:
            where_clauses.append(f"r.race_date <= '{end_date.isoformat()}'")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT
            r.id AS race_id,
            r.race_date,
            r.track_id,
            r.race_number,
            r.track_condition,
            r.weather,
            r.temperature,
            r.humidity,
            r.track_temperature,
            r.grade,
            r.distance,
            r.status AS race_status,
            re.id AS race_entry_id,
            re.rider_id,
            re.post_position,
            re.handicap_meters,
            re.trial_time,
            re.avg_trial_time,
            re.trial_deviation,
            re.race_score,
            re.entry_status,
            rr.finish_position,
            rr.race_time,
            rr.start_timing,
            CAST(rr.accident_code AS TEXT) AS accident_code,
            ri.registration_number AS rider_registration_number,
            ri.generation,
            ri.birth_year,
            ri.home_track_id,
            ri.rank AS rider_rank,
            wo.win_odds,
            wo.win_odds_rank
        FROM race_entries re
        JOIN races r ON re.race_id = r.id
        LEFT JOIN race_results rr ON rr.race_entry_id = re.id
        JOIN riders ri ON re.rider_id = ri.id
        LEFT JOIN (
            SELECT
                os.race_id,
                CAST(os.combination AS INTEGER) AS post_position,
                os.odds_value AS win_odds,
                RANK() OVER (PARTITION BY os.race_id ORDER BY os.odds_value ASC) AS win_odds_rank
            FROM odds_snapshot os
            WHERE os.ticket_type = '単勝'
            AND os.ingestion_batch_id = (
                SELECT MAX(os2.ingestion_batch_id)
                FROM odds_snapshot os2
                WHERE os2.race_id = os.race_id AND os2.ticket_type = '単勝'
            )
        ) wo ON wo.race_id = r.id AND wo.post_position = re.post_position
        {where_sql}
        ORDER BY r.race_date, r.race_number, r.id, re.post_position
        """

        schema_overrides = {
            "race_id": pl.Int64,
            "track_id": pl.Int64,
            "race_number": pl.Int64,
            "track_condition": pl.Utf8,
            "weather": pl.Utf8,
            "temperature": pl.Float64,
            "humidity": pl.Float64,
            "track_temperature": pl.Float64,
            "grade": pl.Utf8,
            "distance": pl.Int64,
            "race_status": pl.Utf8,
            "race_entry_id": pl.Int64,
            "rider_id": pl.Int64,
            "post_position": pl.Int64,
            "handicap_meters": pl.Float64,
            "trial_time": pl.Float64,
            "avg_trial_time": pl.Float64,
            "trial_deviation": pl.Float64,
            "race_score": pl.Float64,
            "entry_status": pl.Utf8,
            "finish_position": pl.Int64,
            "race_time": pl.Float64,
            "start_timing": pl.Float64,
            "accident_code": pl.Utf8,
            "rider_registration_number": pl.Utf8,
            "generation": pl.Int64,
            "birth_year": pl.Int64,
            "home_track_id": pl.Int64,
            "rider_rank": pl.Utf8,
            "win_odds": pl.Float64,
            "win_odds_rank": pl.Int64,
        }

        with self.engine.connect() as conn:
            df = pl.read_database(query=query, connection=conn, schema_overrides=schema_overrides)

        if df.is_empty():
            return df

        if df["race_date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("race_date").str.strptime(pl.Date, strict=False))
        elif df["race_date"].dtype != pl.Date:
            df = df.with_columns(pl.col("race_date").cast(pl.Date))

        return df.with_columns(
            pl.concat_str(
                [
                    pl.col("race_date").cast(pl.Utf8),
                    pl.lit("-"),
                    pl.col("race_number").cast(pl.Utf8),
                    pl.lit("-"),
                    pl.col("post_position").cast(pl.Utf8),
                ]
            ).alias("row_key"),
        )

    def load_all(self) -> pl.DataFrame:
        """Load the full dataset."""
        return self.load_race_dataset()
