#!/usr/bin/env python3
"""Simple backtest for keiba models.

Evaluates win and wide return rates on 2024 test data using
model probabilities vs confirmed payouts.

Usage:
    uv run python scripts/keiba_backtest.py
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import polars as pl
import structlog
from sqlalchemy import create_engine, text

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline
from providence.probability.plackett_luce import compute_all_ticket_probs

logger = structlog.get_logger()


def load_model(surface: str) -> tuple[lgb.Booster, dict]:
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    version_dir = base / version
    model = lgb.Booster(model_file=str(version_dir / "model.txt"))
    metadata = json.loads((version_dir / "metadata.json").read_text())
    return model, metadata


def build_X(features: pl.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns
    return pd.DataFrame({
        col: (
            features[col].cast(pl.Int32).fill_null(-1).to_list()
            if col in cat_cols
            else features[col].cast(pl.Float64).fill_null(float("nan")).to_list()
        )
        for col in feature_columns
    })


def run_backtest(surface: str, features: pl.DataFrame, engine) -> dict:
    log = logger.bind(surface=surface)
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    model, metadata = load_model(surface)
    feature_columns = metadata["feature_columns"]
    temperature = metadata.get("temperature", 1.0)

    race_ids = surface_df["race_id"].unique().sort().to_list()
    log.info("backtest_start", races=len(race_ids), entries=surface_df.shape[0])

    total_win_bet = 0
    total_win_payout = 0
    total_wide_bet = 0
    total_wide_payout = 0
    races_bet_win = 0
    races_bet_wide = 0

    with engine.connect() as conn:
        for race_id in race_ids:
            race_df = surface_df.filter(pl.col("race_id") == race_id).sort("post_position")
            if race_df.shape[0] < 2:
                continue

            X = build_X(race_df, feature_columns)
            scores = model.predict(X)
            probs = compute_all_ticket_probs(scores, temperature)

            post_positions = race_df["post_position"].to_list()
            base_odds_list = race_df["base_win_odds"].to_list()
            race_df["race_key"][0]

            payouts = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(race_id)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts}

            # --- Win backtest ---
            if "win" in probs:
                for idx, (pp, odds) in enumerate(zip(post_positions, base_odds_list)):
                    if odds is None or odds <= 0:
                        continue
                    win_prob = probs["win"].get(idx, 0.0)
                    ev = win_prob * odds
                    if ev > 1.2:
                        bet_amount = 100
                        total_win_bet += bet_amount
                        actual_payout = payout_map.get(("win", str(pp)), 0)
                        if actual_payout > 0:
                            total_win_payout += actual_payout
                        races_bet_win += 1

            # --- Wide backtest ---
            if "wide" in probs:
                for (i, j), prob in sorted(probs["wide"].items(), key=lambda x: -x[1]):
                    if prob < 0.05:
                        continue
                    pp_i = post_positions[i]
                    pp_j = post_positions[j]
                    combo_key = f"{min(pp_i,pp_j):02d}{max(pp_i,pp_j):02d}"
                    wide_payout = payout_map.get(("wide", combo_key), 0)

                    implied_odds = 1.0 / prob if prob > 0 else 0
                    if implied_odds > 0 and prob * implied_odds > 1.0:
                        bet_amount = 100
                        total_wide_bet += bet_amount
                        if wide_payout > 0:
                            payout_per_100 = wide_payout
                            total_wide_payout += payout_per_100
                        races_bet_wide += 1

    win_roi = (total_win_payout / total_win_bet * 100) if total_win_bet > 0 else 0
    wide_roi = (total_wide_payout / total_wide_bet * 100) if total_wide_bet > 0 else 0

    log.info("backtest_result",
             win_bets=races_bet_win, win_bet_total=total_win_bet, win_payout_total=total_win_payout,
             win_roi=f"{win_roi:.1f}%",
             wide_bets=races_bet_wide, wide_bet_total=total_wide_bet, wide_payout_total=total_wide_payout,
             wide_roi=f"{wide_roi:.1f}%")

    return {
        "surface": surface,
        "win_roi": win_roi,
        "wide_roi": wide_roi,
        "win_bets": races_bet_win,
        "wide_bets": races_bet_wide,
    }


def main():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    loader = KeibaDataLoader()
    pipeline = KeibaFeaturePipeline()

    logger.info("loading_test_data")
    t0 = time.time()
    df = loader.load_race_dataset(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))
    logger.info("data_loaded", rows=df.shape[0], seconds=f"{time.time()-t0:.1f}")

    logger.info("building_features")
    t0 = time.time()
    features = pipeline.build_features(df)
    logger.info("features_built", rows=features.shape[0], seconds=f"{time.time()-t0:.1f}")

    engine = create_engine("sqlite:///data/providence.db")

    for surface in ["turf", "dirt"]:
        run_backtest(surface, features, engine)


if __name__ == "__main__":
    main()
