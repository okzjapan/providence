"""Keiba feature pipeline orchestration."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


class KeibaFeaturePipeline:
    """Compute train/predict features for JRA horse racing."""

    categorical_maps = {
        "class_code": {
            "05": 0,
            "10": 1,
            "16": 2,
            "A1": 3,
            "A3": 3,
            "OP": 3,
            "__NULL__": -1,
        },
    }

    excluded_feature_columns = {
        "row_key",
        "race_id",
        "entry_id",
        "race_key",
        "race_date",
        "race_number",
        "blood_registration_number",
        "jockey_code",
        "trainer_code",
        "horse_birth_date",
        "finish_position",
        "race_time_raw",
        "race_time_sec",
        "body_weight_change_raw",
        "confirmed_win_odds",
        "confirmed_popularity",
        "sire_code",
        "broodmare_sire_code",
        "distance_aptitude_code",
        "weight_rule_code",
    }

    def build_features(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        if raw_df.is_empty():
            return raw_df

        df = self._prepare_base(raw_df)
        from providence.keiba.features.field_strength import add_field_strength_features
        from providence.keiba.features.horse import add_horse_features
        from providence.keiba.features.pace import add_pace_features
        from providence.keiba.features.performance import add_performance_features
        from providence.keiba.features.relations import add_relation_features

        df = add_performance_features(df)
        df = add_horse_features(df)
        df = add_pace_features(df)
        df = add_relation_features(df)
        df = add_field_strength_features(df)
        df = self._encode_categoricals(df)
        self.assert_no_leakage(df)
        return df

    def build_features_for_race(
        self, race_entries_df: pl.DataFrame, history_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Build features for one target race using prior history only."""
        if race_entries_df.is_empty():
            return race_entries_df
        features = self.build_features_for_races(race_entries_df, history_df)
        race_id = race_entries_df["race_id"][0]
        return features.filter(pl.col("race_id") == race_id).sort("post_position")

    def build_features_for_races(
        self, race_entries_df: pl.DataFrame, history_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Build features for multiple same-cutoff races in one pass."""
        if race_entries_df.is_empty():
            return race_entries_df

        target_date = race_entries_df["race_date"][0]
        target_race_numbers = tuple(
            int(v) for v in race_entries_df["race_number"].unique().to_list()
        )
        eligible_same_day = history_df.filter(
            (pl.col("race_date") == target_date)
            & (pl.col("race_number") < min(target_race_numbers))
        )
        eligible_before = history_df.filter(pl.col("race_date") < target_date)
        base_history = pl.concat(
            [eligible_before, eligible_same_day], how="vertical_relaxed"
        )

        combined = pl.concat([base_history, race_entries_df], how="vertical_relaxed")
        features = self.build_features(combined)
        target_race_ids = race_entries_df["race_id"].unique(maintain_order=True).to_list()
        return features.filter(pl.col("race_id").is_in(target_race_ids)).sort(
            ["race_number", "race_id", "post_position"]
        )

    @staticmethod
    def history_end_for_date(as_of_date: date) -> date:
        return as_of_date - timedelta(days=1)

    @staticmethod
    def _prepare_base(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(["race_date", "race_number", "race_id", "post_position"])

    @classmethod
    def feature_columns(cls, df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in cls.excluded_feature_columns]

    @classmethod
    def _encode_categoricals(cls, df: pl.DataFrame) -> pl.DataFrame:
        out = df
        for column, mapping in cls.categorical_maps.items():
            if column in out.columns:
                out = out.with_columns(
                    pl.col(column)
                    .cast(pl.Utf8)
                    .fill_null("__NULL__")
                    .replace_strict(mapping, default=-1)
                    .cast(pl.Int32)
                    .alias(column)
                )
        return out

    @classmethod
    def assert_no_leakage(cls, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
        feature_cols = cls.feature_columns(df)
        leaked_in_features = [c for c in feature_cols if c in cls.excluded_feature_columns]
        if leaked_in_features:
            raise ValueError(f"Leaked columns in features: {sorted(leaked_in_features)}")
