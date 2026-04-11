"""Past race detail features (前走データ).

Generates per-runner features from the N most recent completed races.
All lookups are strictly temporal-safe: only races with race_date < current
race_date are used (same-day races are excluded to prevent leakage).
"""

from __future__ import annotations

import numpy as np
import polars as pl

N_PREV_RACES = 5

_PREV_COLS = [
    "finish_position",
    "race_time",
    "start_timing",
    "handicap_meters",
    "track_condition",
    "track_id",
    "win_odds_rank",
    "win_odds",
]


def add_prev_race_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add past-race detail features for each runner.

    For each (rider, race_date) pair, looks up the N most recent prior
    races and extracts per-race metrics plus derived aggregates.
    """
    if df.is_empty():
        return df

    required = {"rider_id", "race_date", "race_id", "finish_position", "race_time", "start_timing"}
    if not required.issubset(df.columns):
        return df

    if "win_odds_rank" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("win_odds_rank"))
    if "win_odds" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("win_odds"))

    history = (
        df.filter(
            pl.col("finish_position").is_not_null()
            & (pl.col("finish_position") >= 1)
            & (pl.col("finish_position") <= 8)
        )
        .select(["rider_id", "race_date", "race_id", "race_number"] + _PREV_COLS)
        .sort(["rider_id", "race_date", "race_number"])
    )

    race_keys = df.select(["race_id", "race_entry_id", "rider_id", "race_date"]).unique(subset=["race_entry_id"])
    result_rows: list[dict] = []

    riders = history["rider_id"].unique().to_list()
    rider_groups = {rid: history.filter(pl.col("rider_id") == rid) for rid in riders}

    for row in race_keys.iter_rows(named=True):
        entry_id = row["race_entry_id"]
        rider_id = row["rider_id"]
        race_date = row["race_date"]

        features: dict = {"race_entry_id": entry_id}

        rider_hist = rider_groups.get(rider_id)
        if rider_hist is None or rider_hist.is_empty():
            result_rows.append(features)
            continue

        prior = rider_hist.filter(pl.col("race_date") < race_date).tail(N_PREV_RACES)

        if prior.is_empty():
            result_rows.append(features)
            continue

        n_available = len(prior)
        positions = prior["finish_position"].to_list()
        race_times = prior["race_time"].to_list()
        start_timings = prior["start_timing"].to_list()
        handicaps = prior["handicap_meters"].to_list()
        tracks = prior["track_id"].to_list()
        conditions = prior["track_condition"].to_list()
        odds_ranks = prior["win_odds_rank"].to_list() if "win_odds_rank" in prior.columns else [None] * n_available
        odds_vals = prior["win_odds"].to_list() if "win_odds" in prior.columns else [None] * n_available

        for i in range(N_PREV_RACES):
            idx = n_available - 1 - i
            suffix = f"_prev{i + 1}"
            if idx >= 0:
                features[f"finish_position{suffix}"] = positions[idx]
                features[f"race_time{suffix}"] = race_times[idx]
                features[f"start_timing{suffix}"] = start_timings[idx]
                features[f"handicap{suffix}"] = handicaps[idx]
                features[f"track_id{suffix}"] = tracks[idx]
                cond = conditions[idx]
                features[f"is_wet{suffix}"] = 1 if cond in ("湿", "重") else (0 if cond is not None else None)
                features[f"odds_rank{suffix}"] = odds_ranks[idx]
                features[f"win_odds{suffix}"] = odds_vals[idx]
                pos = positions[idx]
                orank = odds_ranks[idx]
                if pos is not None and orank is not None:
                    features[f"beat_odds{suffix}"] = 1 if pos < orank else (0 if pos > orank else None)
                else:
                    features[f"beat_odds{suffix}"] = None

        valid_positions = [p for p in positions if p is not None]
        valid_times = [t for t in race_times if t is not None]
        valid_sts = [s for s in start_timings if s is not None]

        if valid_positions:
            weights = np.array([0.5 ** i for i in range(len(valid_positions))])[::-1]
            vp = np.array(valid_positions, dtype=float)
            features["form_score"] = float(np.dot(weights, vp) / weights.sum())

            features["prev_best_finish"] = min(valid_positions)
            features["prev_worst_finish"] = max(valid_positions)
            features["prev_win_count"] = sum(1 for p in valid_positions if p == 1)
            features["prev_top3_count"] = sum(1 for p in valid_positions if p <= 3)

        if valid_times:
            vt = np.array(valid_times, dtype=float)
            weights_t = np.array([0.5 ** i for i in range(len(vt))])[::-1]
            features["ewm_race_time"] = float(np.dot(weights_t, vt) / weights_t.sum())
            features["prev_best_time"] = float(np.min(vt))
            features["prev_time_std"] = float(np.std(vt)) if len(vt) >= 2 else None

        if valid_sts:
            vs = np.array(valid_sts, dtype=float)
            weights_s = np.array([0.5 ** i for i in range(len(vs))])[::-1]
            features["ewm_start_timing"] = float(np.dot(weights_s, vs) / weights_s.sum())

        valid_odds_ranks = [odds_ranks[n_available - 1 - i] for i in range(n_available) if odds_ranks[n_available - 1 - i] is not None]
        if valid_odds_ranks:
            features["avg_odds_rank"] = float(np.mean(valid_odds_ranks))
        beat_count = sum(
            1 for i in range(n_available)
            if positions[i] is not None and odds_ranks[i] is not None and positions[i] < odds_ranks[i]
        )
        total_comparable = sum(
            1 for i in range(n_available)
            if positions[i] is not None and odds_ranks[i] is not None
        )
        features["beat_odds_rate"] = float(beat_count / total_comparable) if total_comparable > 0 else None

        if n_available >= 2:
            recent = valid_positions[-1] if valid_positions else None
            older_avg = np.mean(valid_positions[:-1]) if len(valid_positions) >= 2 else None
            if recent is not None and older_avg is not None:
                features["finish_improvement"] = float(older_avg - recent)

        if valid_positions:
            streak_w, streak_t3, streak_l = 0, 0, 0
            for p in reversed(valid_positions):
                if p == 1:
                    streak_w += 1
                else:
                    break
            for p in reversed(valid_positions):
                if p <= 3:
                    streak_t3 += 1
                else:
                    break
            for p in reversed(valid_positions):
                if p >= 4:
                    streak_l += 1
                else:
                    break
            features["winning_streak"] = streak_w
            features["top3_streak"] = streak_t3
            features["losing_streak"] = streak_l

        if n_available >= 2 and valid_positions:
            vp = np.array(valid_positions[-min(5, len(valid_positions)):], dtype=float)
            if len(vp) >= 2:
                x = np.arange(len(vp), dtype=float)
                features["prev_finish_trend"] = float(-np.polyfit(x, vp, 1)[0])

        if valid_times and len(valid_times) >= 2:
            vt = np.array(valid_times[-min(5, len(valid_times)):], dtype=float)
            x = np.arange(len(vt), dtype=float)
            features["prev_race_time_trend"] = float(-np.polyfit(x, vt, 1)[0])

        if valid_sts and len(valid_sts) >= 2:
            vs = np.array(valid_sts[-min(5, len(valid_sts)):], dtype=float)
            x = np.arange(len(vs), dtype=float)
            features["prev_st_trend"] = float(-np.polyfit(x, vs, 1)[0])

        result_rows.append(features)

    if not result_rows:
        return df

    feat_df = pl.DataFrame(result_rows).cast({"race_entry_id": df["race_entry_id"].dtype})
    return df.join(feat_df, on="race_entry_id", how="left")
