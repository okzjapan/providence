"""Feature pipeline orchestration and caching."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from providence.features.race import add_race_features
from providence.features.rider import add_rider_features
from providence.features.track import add_track_features
from providence.features.trial_run import add_trial_run_features


class FeaturePipeline:
    """Compute train/predict features with temporal safety."""

    categorical_columns = ["track_id", "track_condition", "weather", "grade"]
    categorical_maps = {
        "track_condition": {"良": 0, "湿": 1, "重": 2, "斑": 3, "__NULL__": 4},
        "weather": {"晴": 0, "曇": 1, "雨": 2, "小雨": 3, "小雪": 4, "雪": 5, "other": 6, "__NULL__": 7},
        "grade": {"普通": 0, "GII": 1, "GI": 2, "SG": 3, "__NULL__": 4},
    }
    excluded_feature_columns = {
        "row_key",
        "race_id",
        "race_entry_id",
        "rider_id",
        "rider_registration_number",
        "race_date",
        "race_number",
        "finish_position",
        "race_time",
        "start_timing",
        "accident_code",
        "entry_status",
        "race_status",
        "avg_trial_time",
        "race_score",
        "birth_year",
        "home_track_id",
    }

    def build_features(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        if raw_df.is_empty():
            return raw_df

        df = self._prepare_base(raw_df)
        df = add_trial_run_features(df)
        df = add_rider_features(df)
        df = add_race_features(df)
        df = add_track_features(df)
        df = self._encode_categoricals(df)
        self.assert_no_leakage(df)
        return df

    def build_features_for_race(self, race_entries_df: pl.DataFrame, history_df: pl.DataFrame) -> pl.DataFrame:
        """Build features for one target race using prior history only."""
        if race_entries_df.is_empty():
            return race_entries_df

        target_date = race_entries_df["race_date"][0]
        target_race_number = race_entries_df["race_number"][0]

        eligible_same_day = history_df.filter(
            (pl.col("race_date") == target_date) & (pl.col("race_number") < target_race_number)
        )
        eligible_before = history_df.filter(pl.col("race_date") < target_date)
        base_history = pl.concat([eligible_before, eligible_same_day], how="vertical_relaxed")

        combined = pl.concat([base_history, race_entries_df], how="vertical_relaxed")
        features = self.build_features(combined)
        race_id = race_entries_df["race_id"][0]
        return features.filter(pl.col("race_id") == race_id).sort("post_position")

    def build_and_cache(self, raw_df: pl.DataFrame, cache_path: str) -> pl.DataFrame:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return pl.read_parquet(path)

        features = self.build_features(raw_df)
        features.write_parquet(path)
        return features

    def invalidate_cache(self, cache_glob: str = "data/processed/features_*.parquet") -> int:
        count = 0
        for path in Path(".").glob(cache_glob):
            path.unlink(missing_ok=True)
            count += 1
        return count

    @staticmethod
    def cache_key(stats: dict[str, object]) -> str:
        payload = json.dumps(stats, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _prepare_base(df: pl.DataFrame) -> pl.DataFrame:
        out = df.sort(["race_date", "race_number", "race_id", "post_position"])

        # Normalize rare weather values before integer coding.
        weather_counts = (
            out.group_by("weather")
            .agg(pl.len().alias("n"))
            .filter(pl.col("weather").is_not_null())
            .with_columns(pl.when(pl.col("n") < 20).then(pl.lit("other")).otherwise(pl.col("weather")).alias("mapped"))
            .select(["weather", "mapped"])
        )
        if not weather_counts.is_empty():
            out = out.join(weather_counts, on="weather", how="left").with_columns(
                pl.coalesce(["mapped", "weather"]).alias("weather")
            ).drop("mapped")

        return out

    @classmethod
    def feature_columns(cls, df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in cls.excluded_feature_columns]

    @staticmethod
    def _encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
        out = df
        # track_id is already stable integer-coded
        for column in ("track_condition", "weather", "grade"):
            if column in out.columns:
                mapping = FeaturePipeline.categorical_maps[column]
                out = out.with_columns(
                    pl.col(column)
                    .cast(pl.Utf8)
                    .fill_null("__NULL__")
                    .replace_strict(mapping, default=-1)
                    .cast(pl.Int32)
                    .alias(column)
                )
        return out

    @staticmethod
    def assert_no_leakage(df: pl.DataFrame) -> None:
        """Placeholder assertion boundary.

        Actual leakage verification is tested using targeted synthetic datasets.
        """
        if df.is_empty():
            return
        required = {"race_date", "race_number"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns for leakage checks: {sorted(missing)}")

    @staticmethod
    def history_end_for_date(as_of_date: date) -> date:
        return as_of_date - timedelta(days=1)
