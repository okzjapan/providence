#!/usr/bin/env python3
"""Evaluate market-independent models with multiple Wide strategies.

Combines:
1. Baseline Wide ◎-○ (model top-2)
2. たぬきけいば-style: model top-2 has non-favorite + wide odds >= threshold
3. 冷や飯AI-style: prob^N + EV gap filter
4. Score gap filter: only when model is confident

Usage:
    uv run python scripts/keiba_evaluate_independent.py --surface turf
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import date
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit
from sqlalchemy import create_engine, text

from providence.keiba.features.loader import KeibaDataLoader
from providence.keiba.features.pipeline import KeibaFeaturePipeline


MARKET_CORRELATED_FEATURES = {
    "idm", "jockey_index", "info_index", "training_index",
    "stable_index", "composite_index", "upset_index", "longshot_index",
    "ten_index_pred", "pace_index_pred", "agari_index_pred", "position_index_pred",
    "jockey_index_rank", "composite_rank", "training_rank",
    "idm_rank_in_field", "relative_idm", "gap_to_top",
    "field_avg_idm", "field_max_idm", "field_std_idm",
    "ten_pred_rank", "agari_pred_rank", "position_pred_rank",
}


def load_independent_models(surface: str) -> dict:
    base = Path(f"data/keiba/models/{surface}_independent")
    version = (base / "latest").read_text().strip()
    vdir = base / version
    metadata = json.loads((vdir / "metadata.json").read_text())
    result = {
        "metadata": metadata,
        "feature_columns": metadata["feature_columns"],
        "lambdarank": lgb.Booster(model_file=str(vdir / "lambdarank.txt")),
        "binary_top3": lgb.Booster(model_file=str(vdir / "binary_top3.txt")),
    }
    cal_path = vdir / "calibrator_top3.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            result["calibrator_top3"] = pickle.load(f)
    return result


def build_X(features: pl.DataFrame, feature_columns: list[str], cat_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        col: (features[col].cast(pl.Int32).fill_null(-1).to_list()
              if col in cat_cols
              else features[col].cast(pl.Float64).fill_null(float("nan")).to_list())
        for col in feature_columns
    })


def make_wide_key(pp1: int, pp2: int) -> str:
    a, b = sorted([pp1, pp2])
    return f"{a:02d}{b:02d}"


def evaluate(surface: str, features: pl.DataFrame, engine) -> None:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    models = load_independent_models(surface)
    feat_cols = models["feature_columns"]
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns
    race_ids = surface_df["race_id"].unique().sort().to_list()

    strategies = {}
    strat_names = [
        "win_top1", "wide_top2",
        "tanuki_5pop_w5", "tanuki_5pop_w10",
        "tanuki_4pop_w5", "tanuki_3pop_w8",
        "hiyameshi_p2_gap02", "hiyameshi_p3_gap03",
        "score_gap_wide", "disagree_any",
        "top3_wide_all",
    ]
    for sn in strat_names:
        strategies[sn] = {"bets": 0, "hits": 0, "payout": 0, "monthly": {}}

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            n = race.shape[0]
            if n < 4:
                continue

            X = build_X(race, feat_cols, cat_cols)
            pp_list = race["post_position"].to_list()
            finish_list = race["finish_position"].to_list()
            odds_list = race["confirmed_win_odds"].to_list()
            month_key = str(race["race_date"][0])[:7]

            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            lr_scores = models["lambdarank"].predict(X)
            bt3_raw = models["binary_top3"].predict(X)
            if "calibrator_top3" in models:
                bt3_probs = np.array(models["calibrator_top3"].predict(bt3_raw))
            else:
                bt3_probs = expit(bt3_raw)
            bt3_probs = bt3_probs / (bt3_probs.sum() + 1e-12)

            model_order = np.argsort(-lr_scores)
            top1_idx, top2_idx = int(model_order[0]), int(model_order[1])
            top3_idx = int(model_order[2]) if n > 2 else top2_idx

            valid_odds = [(i, o) for i, o in enumerate(odds_list) if o and o > 0]
            if len(valid_odds) < 4:
                continue
            odds_arr = np.full(n, 999.0)
            for i, o in valid_odds:
                odds_arr[i] = o
            market_order = np.argsort(odds_arr)
            market_rank = np.argsort(np.argsort(odds_arr))

            def record(sn, pp1, pp2, ticket="wide"):
                key = make_wide_key(pp1, pp2)
                s = strategies[sn]
                s["bets"] += 1
                if month_key not in s["monthly"]:
                    s["monthly"][month_key] = {"bet": 0, "payout": 0}
                s["monthly"][month_key]["bet"] += 100
                p = payout_map.get((ticket, key), 0)
                if p > 0:
                    s["hits"] += 1
                    s["payout"] += p
                    s["monthly"][month_key]["payout"] += p

            def record_win(sn, pp):
                s = strategies[sn]
                s["bets"] += 1
                if month_key not in s["monthly"]:
                    s["monthly"][month_key] = {"bet": 0, "payout": 0}
                s["monthly"][month_key]["bet"] += 100
                p = payout_map.get(("win", str(pp)), 0)
                if p > 0:
                    s["hits"] += 1
                    s["payout"] += p
                    s["monthly"][month_key]["payout"] += p

            pp1, pp2 = pp_list[top1_idx], pp_list[top2_idx]

            record_win("win_top1", pp1)
            record("wide_top2", pp1, pp2)

            # Top-3 all Wide (3 combos)
            for i, j in combinations([top1_idx, top2_idx, top3_idx], 2):
                record("top3_wide_all", pp_list[i], pp_list[j])

            # たぬきけいば variants: top-2 has non-favorite horse
            # Use confirmed_win_odds of the non-favorite as proxy for Wide payout potential
            top1_mrank = int(market_rank[top1_idx])
            top2_mrank = int(market_rank[top2_idx])
            nonfav_odds = max(odds_arr[top1_idx], odds_arr[top2_idx])

            has_nonfav_5 = top1_mrank >= 4 or top2_mrank >= 4
            has_nonfav_4 = top1_mrank >= 3 or top2_mrank >= 3
            has_nonfav_3 = top1_mrank >= 2 or top2_mrank >= 2

            if has_nonfav_5:
                record("tanuki_5pop_w5", pp1, pp2)
            if has_nonfav_5 and nonfav_odds >= 15:
                record("tanuki_5pop_w10", pp1, pp2)
            if has_nonfav_4:
                record("tanuki_4pop_w5", pp1, pp2)
            if has_nonfav_3 and nonfav_odds >= 10:
                record("tanuki_3pop_w8", pp1, pp2)

            # Disagree: model top-2 != market top-2
            market_top2 = set(market_order[:2].tolist())
            model_top2 = {top1_idx, top2_idx}
            if model_top2 != market_top2:
                record("disagree_any", pp1, pp2)

            # 冷や飯 style: prob^N × odds, EV gap
            win_probs = bt3_probs
            for power, gap_thresh, sn in [(2, 0.2, "hiyameshi_p2_gap02"), (3, 0.3, "hiyameshi_p3_gap03")]:
                powered = win_probs ** power
                evs = np.array([powered[i] * odds_arr[i] for i in range(n)])
                ev_sorted = np.sort(evs)[::-1]
                if len(ev_sorted) >= 2 and (ev_sorted[0] - ev_sorted[1]) >= gap_thresh:
                    best_idx = int(np.argmax(evs))
                    record_win(sn, pp_list[best_idx])

            # Score gap filter: model's 1st-2nd gap is large
            score_range = lr_scores.max() - lr_scores.min()
            if score_range > 0:
                norm_gap = (lr_scores[top1_idx] - lr_scores[top2_idx]) / score_range
                if norm_gap >= 0.2:
                    record("score_gap_wide", pp1, pp2)

    print(f"\n{'=' * 75}")
    print(f"  {surface.upper()} — Market-Independent Model Evaluation")
    print(f"  Features: {len(feat_cols)} (excluded {len(MARKET_CORRELATED_FEATURES)} market features)")
    print(f"{'=' * 75}")

    for sn in strat_names:
        s = strategies[sn]
        if s["bets"] == 0:
            continue
        roi = s["payout"] / (s["bets"] * 100) * 100
        hit = s["hits"] / s["bets"] * 100
        avg_p = s["payout"] / s["hits"] if s["hits"] else 0
        mrs = []
        for m, d in sorted(s["monthly"].items()):
            if d["bet"] > 0:
                mrs.append(d["payout"] / d["bet"] * 100)
        sharpe = (np.mean(mrs) - 100) / np.std(mrs) if len(mrs) > 1 and np.std(mrs) > 0 else 0

        h1p = sum(d["payout"] for m, d in s["monthly"].items() if m < "2024-07")
        h1b = sum(d["bet"] for m, d in s["monthly"].items() if m < "2024-07")
        h2p = sum(d["payout"] for m, d in s["monthly"].items() if m >= "2024-07")
        h2b = sum(d["bet"] for m, d in s["monthly"].items() if m >= "2024-07")

        h1_roi = h1p / h1b * 100 if h1b else 0
        h2_roi = h2p / h2b * 100 if h2b else 0

        print(f"\n  [{sn}]")
        print(f"  ROI: {roi:.1f}% | {s['hits']}/{s['bets']} ({hit:.1f}%) | avg ¥{avg_p:.0f} | Sharpe: {sharpe:.2f}")
        print(f"  H1: {h1_roi:.1f}% | H2: {h2_roi:.1f}%")


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

    for sname, scode in surfaces.items():
        loader = KeibaDataLoader()
        pipeline = KeibaFeaturePipeline()
        print(f"\n[{sname.upper()}] Loading all data...")
        t0 = time.time()
        df = loader.load_race_dataset(end_date=eval_end)
        sdf = df.filter(pl.col("surface_code") == scode)
        print(f"[{sname.upper()}] Building features ({sdf.shape[0]} rows)...")
        features = pipeline.build_features(sdf)
        features = features.filter(
            (pl.col("race_date") >= eval_start) & (pl.col("race_date") <= eval_end)
        )
        print(f"[{sname.upper()}] Ready in {time.time()-t0:.0f}s → {features.shape[0]} rows")
        evaluate(sname, features, engine)


if __name__ == "__main__":
    main()
