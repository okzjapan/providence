#!/usr/bin/env python3
"""Value Wide strategy: bet on combinations where the model disagrees with the market.

Instead of simple ◎-○ (top-2 by model), this strategy:
1. Picks the model's Top-1 as the anchor
2. Finds a "value" horse: model ranks it much higher than the market does
3. Constructs Wide anchor + value_horse

Usage:
    uv run python scripts/keiba_evaluate_value_wide.py --surface turf
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit
from sqlalchemy import create_engine, text

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline
from providence.probability.plackett_luce import compute_all_ticket_probs


def load_models(surface: str) -> dict:
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    version_dir = base / version
    metadata = json.loads((version_dir / "metadata.json").read_text())
    result = {"metadata": metadata, "feature_columns": metadata["feature_columns"]}
    result["lambdarank"] = lgb.Booster(model_file=str(version_dir / "lambdarank.txt"))
    if metadata.get("model_type") == "phase3c_ensemble":
        result["binary_top3"] = lgb.Booster(model_file=str(version_dir / "binary_top3.txt"))
        cal_path = version_dir / "calibrator_top3.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                result["calibrator_top3"] = pickle.load(f)
    return result


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


def make_wide_key(pp1: int, pp2: int) -> str:
    a, b = sorted([pp1, pp2])
    return f"{a:02d}{b:02d}"


def evaluate_value_wide(surface: str, features: pl.DataFrame, engine) -> None:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    models = load_models(surface)
    feature_columns = models["feature_columns"]
    temperature = models["metadata"].get("temperature", 1.0)
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns
    race_ids = surface_df["race_id"].unique().sort().to_list()

    strategies = {
        "baseline_top2": {"bets": 0, "hits": 0, "payout": 0, "monthly": {}},
        "value_top1_plus_value": {"bets": 0, "hits": 0, "payout": 0, "monthly": {}},
        "value_top1_plus_value_strict": {"bets": 0, "hits": 0, "payout": 0, "monthly": {}},
        "top3_model_all_wide": {"bets": 0, "hits": 0, "payout": 0, "monthly": {}},
        "disagree_wide": {"bets": 0, "hits": 0, "payout": 0, "monthly": {}},
    }

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            n = race.shape[0]
            if n < 4:
                continue

            X = build_X(race, feature_columns, cat_cols)
            pp_list = race["post_position"].to_list()
            finish_list = race["finish_position"].to_list()
            odds_list = race["confirmed_win_odds"].to_list()
            race_date_str = str(race["race_date"][0])
            month_key = race_date_str[:7]

            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            lr_scores = models["lambdarank"].predict(X)
            model_rank = np.argsort(np.argsort(-lr_scores))

            valid_odds = [(i, o) for i, o in enumerate(odds_list) if o and o > 0]
            if len(valid_odds) < 4:
                continue
            odds_arr = np.array([999.0] * n)
            for i, o in valid_odds:
                odds_arr[i] = o
            market_rank = np.argsort(np.argsort(odds_arr))

            top1_model = int(np.argmax(lr_scores))

            def record_bet(strat_name, pp1, pp2):
                key = make_wide_key(pp1, pp2)
                s = strategies[strat_name]
                s["bets"] += 1
                if month_key not in s["monthly"]:
                    s["monthly"][month_key] = {"bet": 0, "payout": 0}
                s["monthly"][month_key]["bet"] += 100
                payout = payout_map.get(("wide", key), 0)
                if payout > 0:
                    s["hits"] += 1
                    s["payout"] += payout
                    s["monthly"][month_key]["payout"] += payout

            # Baseline: model top-2
            top2_model = np.argsort(-lr_scores)[:2]
            record_bet("baseline_top2", pp_list[top2_model[0]], pp_list[top2_model[1]])

            # Value strategy: top-1 + best "value" horse
            # Value = model ranks higher than market
            rank_diff = market_rank - model_rank
            rank_diff[top1_model] = -999
            value_idx = int(np.argmax(rank_diff))
            if rank_diff[value_idx] >= 2:
                record_bet("value_top1_plus_value", pp_list[top1_model], pp_list[value_idx])

            # Strict value: rank_diff >= 4
            if rank_diff[value_idx] >= 4:
                record_bet("value_top1_plus_value_strict", pp_list[top1_model], pp_list[value_idx])

            # Top-3 model: all 3 Wide combinations from model's top 3
            top3_model = np.argsort(-lr_scores)[:3]
            from itertools import combinations
            for i, j in combinations(top3_model, 2):
                record_bet("top3_model_all_wide", pp_list[i], pp_list[j])

            # Disagree Wide: find horse where model_rank <= 3 but market_rank >= 5
            disagree_horses = [i for i in range(n) if model_rank[i] <= 2 and market_rank[i] >= 4]
            if disagree_horses:
                best_disagree = max(disagree_horses, key=lambda i: lr_scores[i])
                if best_disagree != top1_model:
                    record_bet("disagree_wide", pp_list[top1_model], pp_list[best_disagree])

    print(f"\n{'=' * 70}")
    print(f"  {surface.upper()} — Value Wide Strategy Evaluation")
    print(f"{'=' * 70}")

    for name, s in strategies.items():
        wb, wh, wp = s["bets"], s["hits"], s["payout"]
        if wb == 0:
            continue
        roi = wp / (wb * 100) * 100
        hit_rate = wh / wb * 100
        avg_payout = wp / wh if wh else 0

        monthly_rois = []
        for m, d in sorted(s["monthly"].items()):
            if d["bet"] > 0:
                monthly_rois.append(d["payout"] / d["bet"] * 100)
        sharpe = (
            (np.mean(monthly_rois) - 100) / np.std(monthly_rois)
            if len(monthly_rois) > 1 and np.std(monthly_rois) > 0 else 0
        )

        h1_payout = sum(d["payout"] for m, d in s["monthly"].items() if m < "2024-07")
        h1_bet = sum(d["bet"] for m, d in s["monthly"].items() if m < "2024-07")
        h2_payout = sum(d["payout"] for m, d in s["monthly"].items() if m >= "2024-07")
        h2_bet = sum(d["bet"] for m, d in s["monthly"].items() if m >= "2024-07")

        print(f"\n  [{name}]")
        print(f"  Wide ROI:    {roi:.1f}% | {wh}/{wb} ({hit_rate:.1f}%) | avg ¥{avg_payout:.0f}")
        print(f"  Sharpe:      {sharpe:.2f}")
        print(f"  H1/H2 ROI:  {h1_payout/(h1_bet)*100:.1f}% / {h2_payout/(h2_bet)*100:.1f}%" if h1_bet and h2_bet else "  H1/H2: N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    eval_start = date.fromisoformat(args.start_date)
    eval_end = date.fromisoformat(args.end_date)
    engine = create_engine("sqlite:///data/providence.db")
    surfaces = (
        {"turf": 1, "dirt": 2} if args.surface == "both" else {args.surface: (1 if args.surface == "turf" else 2)}
    )

    for surface_name, surface_code in surfaces.items():
        loader = KeibaDataLoader()
        pipeline = KeibaFeaturePipeline()
        print(f"\n[{surface_name.upper()}] Loading all data...")
        t0 = time.time()
        df = loader.load_race_dataset(end_date=eval_end)
        surface_df = df.filter(pl.col("surface_code") == surface_code)
        print(f"[{surface_name.upper()}] Building features ({surface_df.shape[0]} rows)...")
        features = pipeline.build_features(surface_df)
        features = features.filter(
            (pl.col("race_date") >= eval_start) & (pl.col("race_date") <= eval_end)
        )
        print(f"[{surface_name.upper()}] Ready in {time.time() - t0:.0f}s → {features.shape[0]} eval rows")
        evaluate_value_wide(surface_name, features, engine)


if __name__ == "__main__":
    main()
