#!/usr/bin/env python3
"""Betting strategy layer: selective betting on top of model predictions.

Instead of betting every race, apply filters to select only high-value races.
This is where profit is generated - the model identifies horses, the strategy
identifies WHEN to bet.

Usage:
    uv run python scripts/keiba_betting_strategy.py --surface turf --year 2024
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


def load_models(surface: str) -> dict:
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    vdir = base / version
    metadata = json.loads((vdir / "metadata.json").read_text())
    result = {"metadata": metadata, "feature_columns": metadata["feature_columns"]}
    result["lambdarank"] = lgb.Booster(model_file=str(vdir / "lambdarank.txt"))
    result["binary_top3"] = lgb.Booster(model_file=str(vdir / "binary_top3.txt"))
    result["binary_win"] = lgb.Booster(model_file=str(vdir / "binary_win.txt"))
    huber_path = vdir / "huber.txt"
    if huber_path.exists():
        result["huber"] = lgb.Booster(model_file=str(huber_path))
    for name in ("calibrator_win", "calibrator_top3"):
        p = vdir / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                result[name] = pickle.load(f)
    return result


def build_X(features: pl.DataFrame, feat_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        c: (features[c].cast(pl.Int32).fill_null(-1).to_list() if c in cat_cols
            else features[c].cast(pl.Float64).fill_null(float("nan")).to_list())
        for c in feat_cols
    })


def make_wide_key(pp1: int, pp2: int) -> str:
    a, b = sorted([pp1, pp2])
    return f"{a:02d}{b:02d}"


def evaluate_strategies(surface: str, features: pl.DataFrame, engine, year: int) -> None:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    models = load_models(surface)
    feat_cols = models["feature_columns"]
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns
    race_ids = surface_df["race_id"].unique().sort().to_list()

    # All strategies
    strats: dict[str, dict] = {}
    snames = [
        "all_wide_lr", "all_wide_bt3", "all_wide_huber", "all_wide_ensemble",
        "all_win_lr",
        # Filters
        "f_score_gap_20_wide", "f_score_gap_30_wide",
        "f_ev_gap_02_win", "f_ev_gap_03_win",
        "f_disagree_top1_wide", "f_disagree_top2_wide",
        "f_nonfav_in_top2_wide",
        "f_bt3_prob_high_wide",
        "f_combined_a_wide", "f_combined_b_wide",
        "f_combined_c_win",
    ]
    for s in snames:
        strats[s] = {"bets": 0, "hits": 0, "payout": 0, "monthly": {}}

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

            # Model predictions
            lr_scores = models["lambdarank"].predict(X)
            bt3_raw = models["binary_top3"].predict(X)
            bt3_probs = expit(bt3_raw)
            if "calibrator_top3" in models:
                bt3_probs = np.array(models["calibrator_top3"].predict(bt3_raw))
            bt3_norm = bt3_probs / (bt3_probs.sum() + 1e-12)

            bw_raw = models["binary_win"].predict(X)
            if "calibrator_win" in models:
                bw_probs = np.array(models["calibrator_win"].predict(bw_raw))
            else:
                bw_probs = expit(bw_raw)
            bw_norm = bw_probs / (bw_probs.sum() + 1e-12)

            huber_scores = models["huber"].predict(X) if "huber" in models else lr_scores

            # Ensemble
            lr_n = (lr_scores - lr_scores.min()) / (lr_scores.max() - lr_scores.min() + 1e-12)
            h_n = (huber_scores - huber_scores.min()) / (huber_scores.max() - huber_scores.min() + 1e-12)
            ens_scores = 0.3 * lr_n + 0.1 * bw_norm + 0.4 * bt3_norm + 0.2 * h_n

            # Market ranks
            valid_odds = [(i, o) for i, o in enumerate(odds_list) if o and o > 0]
            if len(valid_odds) < 4:
                continue
            odds_arr = np.full(n, 999.0)
            for i, o in valid_odds:
                odds_arr[i] = o
            market_rank = np.argsort(np.argsort(odds_arr))

            # Helpers
            def record(sn, pp1, pp2=None, ticket="wide"):
                s = strats[sn]
                s["bets"] += 1
                if month_key not in s["monthly"]:
                    s["monthly"][month_key] = {"bet": 0, "payout": 0}
                s["monthly"][month_key]["bet"] += 100
                if pp2 is not None:
                    key = make_wide_key(pp1, pp2)
                    p = payout_map.get((ticket, key), 0)
                else:
                    p = payout_map.get(("win", str(pp1)), 0)
                if p > 0:
                    s["hits"] += 1
                    s["payout"] += p
                    s["monthly"][month_key]["payout"] += p

            # Model top-2 for each
            lr_top = np.argsort(-lr_scores)[:2]
            bt3_top = np.argsort(-bt3_probs)[:2]
            hub_top = np.argsort(-huber_scores)[:2]
            ens_top = np.argsort(-ens_scores)[:2]

            lr_pp1, lr_pp2 = pp_list[lr_top[0]], pp_list[lr_top[1]]
            bt3_pp1, bt3_pp2 = pp_list[bt3_top[0]], pp_list[bt3_top[1]]
            hub_pp1, hub_pp2 = pp_list[hub_top[0]], pp_list[hub_top[1]]
            ens_pp1, ens_pp2 = pp_list[ens_top[0]], pp_list[ens_top[1]]

            # --- Baseline: all races ---
            record("all_wide_lr", lr_pp1, lr_pp2)
            record("all_wide_bt3", bt3_pp1, bt3_pp2)
            record("all_wide_huber", hub_pp1, hub_pp2)
            record("all_wide_ensemble", ens_pp1, ens_pp2)
            record("all_win_lr", lr_pp1)

            # --- Filter: score gap (LR 1st-2nd gap / range) ---
            score_range = lr_scores.max() - lr_scores.min()
            if score_range > 0:
                gap = (lr_scores[lr_top[0]] - lr_scores[lr_top[1]]) / score_range
                if gap >= 0.20:
                    record("f_score_gap_20_wide", lr_pp1, lr_pp2)
                if gap >= 0.30:
                    record("f_score_gap_30_wide", lr_pp1, lr_pp2)

            # --- Filter: EV gap (冷や飯 style, win probs) ---
            evs = bw_norm * odds_arr
            ev_sorted = np.sort(evs)[::-1]
            if len(ev_sorted) >= 2:
                ev_gap = ev_sorted[0] - ev_sorted[1]
                best_ev_idx = int(np.argmax(evs))
                if ev_gap >= 0.2:
                    record("f_ev_gap_02_win", pp_list[best_ev_idx])
                if ev_gap >= 0.3:
                    record("f_ev_gap_03_win", pp_list[best_ev_idx])

            # --- Filter: model disagrees with market on top-1 ---
            model_top1 = int(np.argmax(ens_scores))
            market_top1 = int(np.argmin(odds_arr))
            if model_top1 != market_top1:
                record("f_disagree_top1_wide", ens_pp1, ens_pp2)

            # --- Filter: model top-2 differs from market top-2 ---
            model_top2_set = set(ens_top[:2].tolist())
            market_top2_set = set(np.argsort(odds_arr)[:2].tolist())
            if model_top2_set != market_top2_set:
                record("f_disagree_top2_wide", ens_pp1, ens_pp2)

            # --- Filter: non-favorite (4th+ by market) in model top-2 ---
            top2_market_ranks = [int(market_rank[ens_top[0]]), int(market_rank[ens_top[1]])]
            if max(top2_market_ranks) >= 3:
                record("f_nonfav_in_top2_wide", ens_pp1, ens_pp2)

            # --- Filter: BT3 top-1 prob >= 40% (high confidence top-3 pick) ---
            if bt3_norm[bt3_top[0]] >= 0.40:
                record("f_bt3_prob_high_wide", bt3_pp1, bt3_pp2)

            # --- Combined A: score gap >= 0.20 AND 3-5x top pick odds ---
            top_odds = odds_arr[lr_top[0]]
            if score_range > 0 and gap >= 0.20 and 3 <= top_odds < 10:
                record("f_combined_a_wide", lr_pp1, lr_pp2)

            # --- Combined B: ensemble disagrees with market top-2 AND confidence mid+ ---
            ens_range = ens_scores.max() - ens_scores.min()
            if ens_range > 0:
                ens_gap = (ens_scores[ens_top[0]] - ens_scores[ens_top[1]]) / ens_range
            else:
                ens_gap = 0
            if model_top2_set != market_top2_set and ens_gap >= 0.10:
                record("f_combined_b_wide", ens_pp1, ens_pp2)

            # --- Combined C: EV gap >= 0.2 AND model top-1 not market favorite ---
            if ev_gap >= 0.2 and model_top1 != market_top1:
                record("f_combined_c_win", pp_list[best_ev_idx])

    print(f"\n{'=' * 80}")
    print(f"  {surface.upper()} {year} — Betting Strategy Evaluation (v006, 140 features, 4 models)")
    print(f"{'=' * 80}")
    print(f"  {'Strategy':<30s} {'Bets':>6s} {'Hits':>5s} {'Hit%':>6s} {'AvgPay':>7s} {'ROI':>8s} {'Sharpe':>7s}")
    print(f"  {'-'*72}")

    for sn in snames:
        s = strats[sn]
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

        marker = " ★" if roi >= 100 else " ●" if roi >= 90 else ""
        print(f"  {sn:<30s} {s['bets']:>6d} {s['hits']:>5d} {hit:>5.1f}% {avg_p:>6.0f}  {roi:>6.1f}%  {sharpe:>6.2f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    eval_start = date(args.year, 1, 1)
    eval_end = date(args.year, 12, 31)
    engine = create_engine("sqlite:///data/providence.db")

    surfaces = {"turf": 1, "dirt": 2} if args.surface == "both" else {args.surface: (1 if args.surface == "turf" else 2)}

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
        evaluate_strategies(sname, features, engine, args.year)


if __name__ == "__main__":
    main()
