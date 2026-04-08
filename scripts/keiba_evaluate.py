#!/usr/bin/env python3
"""Unified evaluation script for keiba model improvements.

Computes consistent before/after metrics across all Phase 3b steps.
All ROI calculations use confirmed payouts from ticket_payouts table.

Usage:
    uv run python scripts/keiba_evaluate.py
    uv run python scripts/keiba_evaluate.py --surface turf
    uv run python scripts/keiba_evaluate.py --ev-threshold 1.2
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline
from providence.probability.plackett_luce import compute_all_ticket_probs


def load_model(surface: str) -> tuple[lgb.Booster, dict]:
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    model = lgb.Booster(model_file=str(base / version / "model.txt"))
    metadata = json.loads((base / version / "metadata.json").read_text())
    return model, metadata


def build_X(features: pl.DataFrame, feature_columns: list[str], cat_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: (
                features[col].cast(pl.Int32).fill_null(-1).to_list()
                if col in cat_cols
                else features[col].cast(pl.Float64).fill_null(float("nan")).to_list()
            )
            for col in feature_columns
        }
    )


def evaluate_surface(
    surface: str,
    features: pl.DataFrame,
    engine,
    *,
    ev_threshold: float = 1.2,
    calibrator=None,
) -> dict:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    model, metadata = load_model(surface)
    feature_columns = metadata["feature_columns"]
    temperature = metadata.get("temperature", 1.0)
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns

    race_ids = surface_df["race_id"].unique().sort().to_list()

    top_pick_bets = 0
    top_pick_wins = 0
    top_pick_payout = 0

    ev_bets = 0
    ev_wins = 0
    ev_payout = 0

    monthly_roi: dict[str, dict] = {}
    all_brier_preds = []
    all_brier_actuals = []

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            if race.shape[0] < 2:
                continue

            X = build_X(race, feature_columns, cat_cols)
            scores = model.predict(X)
            probs = compute_all_ticket_probs(scores, temperature)

            win_probs = [probs["win"].get(i, 0.0) for i in range(race.shape[0])]

            if calibrator is not None:
                win_probs = calibrator.predict(np.array(win_probs)).tolist()
                total = sum(win_probs)
                if total > 0:
                    win_probs = [p / total for p in win_probs]

            odds_list = race["base_win_odds"].to_list()
            finish_list = race["finish_position"].to_list()
            pp_list = race["post_position"].to_list()
            race_date_str = str(race["race_date"][0])
            month_key = race_date_str[:7]

            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            if month_key not in monthly_roi:
                monthly_roi[month_key] = {"bet": 0, "payout": 0}

            best_idx = int(np.argmax(scores))
            pp_best = pp_list[best_idx]
            actual_payout = payout_map.get(("win", str(pp_best)), 0)
            top_pick_bets += 1
            monthly_roi[month_key]["bet"] += 100
            if actual_payout > 0:
                top_pick_wins += 1
                top_pick_payout += actual_payout
                monthly_roi[month_key]["payout"] += actual_payout

            for idx in range(race.shape[0]):
                wp = win_probs[idx]
                odds = odds_list[idx]
                actual_win = 1 if finish_list[idx] == 1 else 0
                all_brier_preds.append(wp)
                all_brier_actuals.append(actual_win)

                if odds and odds > 0:
                    ev = wp * odds
                    if ev > ev_threshold:
                        ev_bets += 1
                        pp = pp_list[idx]
                        ev_actual = payout_map.get(("win", str(pp)), 0)
                        if ev_actual > 0:
                            ev_wins += 1
                            ev_payout += ev_actual

    top_pick_roi = top_pick_payout / (top_pick_bets * 100) * 100 if top_pick_bets else 0
    ev_roi = ev_payout / (ev_bets * 100) * 100 if ev_bets else 0
    brier = np.mean([(p - a) ** 2 for p, a in zip(all_brier_preds, all_brier_actuals)]) if all_brier_preds else 0

    monthly_rois = []
    for m, data in sorted(monthly_roi.items()):
        if data["bet"] > 0:
            monthly_rois.append(data["payout"] / data["bet"] * 100)
    sharpe = (
        (np.mean(monthly_rois) - 100) / np.std(monthly_rois)
        if len(monthly_rois) > 1 and np.std(monthly_rois) > 0
        else 0
    )

    sum(1 for m in monthly_roi if m < "2024-07")
    h1_payout = sum(d["payout"] for m, d in monthly_roi.items() if m < "2024-07")
    h1_total_bet = sum(d["bet"] for m, d in monthly_roi.items() if m < "2024-07")
    h2_payout = sum(d["payout"] for m, d in monthly_roi.items() if m >= "2024-07")
    h2_total_bet = sum(d["bet"] for m, d in monthly_roi.items() if m >= "2024-07")

    return {
        "surface": surface,
        "top_pick_roi": round(top_pick_roi, 1),
        "top_pick_wins": top_pick_wins,
        "top_pick_bets": top_pick_bets,
        "top_pick_hit_rate": round(top_pick_wins / top_pick_bets * 100, 1) if top_pick_bets else 0,
        "ev_filter_roi": round(ev_roi, 1),
        "ev_bets": ev_bets,
        "ev_wins": ev_wins,
        "ev_hit_rate": round(ev_wins / ev_bets * 100, 1) if ev_bets else 0,
        "brier_score": round(brier, 4),
        "sharpe_ratio": round(sharpe, 2),
        "h1_roi": round(h1_payout / h1_total_bet * 100, 1) if h1_total_bet else 0,
        "h2_roi": round(h2_payout / h2_total_bet * 100, 1) if h2_total_bet else 0,
        "feature_count": len(feature_columns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    parser.add_argument("--ev-threshold", type=float, default=1.2)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    loader = KeibaDataLoader()
    pipeline = KeibaFeaturePipeline()

    print("Loading data...")
    t0 = time.time()
    df = loader.load_race_dataset(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
    )
    print(f"Building features ({df.shape[0]} rows)...")
    features = pipeline.build_features(df)
    print(f"Features ready in {time.time() - t0:.0f}s ({features.shape[1]} cols)")

    engine = create_engine("sqlite:///data/providence.db")
    surfaces = (
        {"turf": 1, "dirt": 2} if args.surface == "both" else {args.surface: (1 if args.surface == "turf" else 2)}
    )

    for surface_name in surfaces:
        result = evaluate_surface(surface_name, features, engine, ev_threshold=args.ev_threshold)
        print(f"\n{'=' * 50}")
        print(f"  {surface_name.upper()} (EV threshold={args.ev_threshold})")
        print(f"{'=' * 50}")
        print(f"  Features:        {result['feature_count']}")
        tp = result
        print(f"  Top Pick ROI:    {tp['top_pick_roi']}% ({tp['top_pick_wins']}/{tp['top_pick_bets']} wins)")
        print(f"  EV Filter ROI:   {tp['ev_filter_roi']}% ({tp['ev_wins']}/{tp['ev_bets']} wins)")
        print(f"  Brier Score:     {result['brier_score']}")
        print(f"  Sharpe Ratio:    {result['sharpe_ratio']}")
        print(f"  H1 ROI (Jan-Jun): {result['h1_roi']}%")
        print(f"  H2 ROI (Jul-Dec): {result['h2_roi']}%")


if __name__ == "__main__":
    main()
