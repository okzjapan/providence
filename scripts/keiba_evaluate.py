#!/usr/bin/env python3
"""Unified evaluation script for keiba model improvements.

Phase 3c: Confirmed-odds based evaluation with overlay/divergence analysis.
Supports both single LambdaRank models and Phase 3c multi-model ensembles.

Usage:
    uv run python scripts/keiba_evaluate.py
    uv run python scripts/keiba_evaluate.py --surface turf
    uv run python scripts/keiba_evaluate.py --ev-threshold 1.2
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

JRA_RAKE = 0.20
ODDS_BANDS = [
    ("1-3x", 1.0, 3.0),
    ("3-5x", 3.0, 5.0),
    ("5-10x", 5.0, 10.0),
    ("10-20x", 10.0, 20.0),
    ("20-50x", 20.0, 50.0),
    ("50x+", 50.0, 9999.0),
]
OVERLAY_THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.50]


def load_phase3c_models(surface: str) -> dict:
    """Load Phase 3c models (LambdaRank + BinaryWin + BinaryTop3 + calibrators)."""
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    version_dir = base / version
    metadata = json.loads((version_dir / "metadata.json").read_text())

    result = {"metadata": metadata, "feature_columns": metadata["feature_columns"]}

    if metadata.get("model_type") == "phase3c_ensemble":
        result["lambdarank"] = lgb.Booster(model_file=str(version_dir / "lambdarank.txt"))
        result["binary_win"] = lgb.Booster(model_file=str(version_dir / "binary_win.txt"))
        result["binary_top3"] = lgb.Booster(model_file=str(version_dir / "binary_top3.txt"))

        cal_win_path = version_dir / "calibrator_win.pkl"
        cal_top3_path = version_dir / "calibrator_top3.pkl"
        if cal_win_path.exists():
            with open(cal_win_path, "rb") as f:
                result["calibrator_win"] = pickle.load(f)
        if cal_top3_path.exists():
            with open(cal_top3_path, "rb") as f:
                result["calibrator_top3"] = pickle.load(f)

        result["is_ensemble"] = True
    else:
        result["lambdarank"] = lgb.Booster(model_file=str(version_dir / "model.txt"))
        result["is_ensemble"] = False

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


def market_implied_prob(confirmed_odds: float) -> float:
    if confirmed_odds is None or confirmed_odds <= 0:
        return 0.0
    return (1.0 - JRA_RAKE) / confirmed_odds


def normalize_within_race(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    n = len(probs)
    return [1.0 / n] * n if n > 0 else probs


def evaluate_surface(
    surface: str,
    features: pl.DataFrame,
    engine,
    *,
    ev_threshold: float = 1.2,
) -> dict:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    models = load_phase3c_models(surface)
    feature_columns = models["feature_columns"]
    temperature = models["metadata"].get("temperature", 1.0)
    is_ensemble = models["is_ensemble"]
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns

    race_ids = surface_df["race_id"].unique().sort().to_list()

    model_names = ["lambdarank"]
    if is_ensemble:
        model_names.extend(["binary_win", "binary_top3"])

    stats: dict[str, dict] = {}
    for mn in model_names:
        stats[mn] = {
            "top_pick_bets": 0, "top_pick_wins": 0, "top_pick_payout": 0,
            "ev_bets": 0, "ev_wins": 0, "ev_payout": 0,
            "monthly_roi": {},
            "brier_preds": [], "brier_actuals": [],
            "overlay_bets": {t: [] for t in OVERLAY_THRESHOLDS},
            "odds_band_stats": {
                name: {"model_probs": [], "actual_wins": []}
                for name, _, _ in ODDS_BANDS
            },
            "top_pick_by_band": {
                name: {"bets": 0, "wins": 0, "payout": 0}
                for name, _, _ in ODDS_BANDS
            },
        }

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            if race.shape[0] < 2:
                continue

            X = build_X(race, feature_columns, cat_cols)
            confirmed_odds_list = race["confirmed_win_odds"].to_list()
            finish_list = race["finish_position"].to_list()
            pp_list = race["post_position"].to_list()
            race_date_str = str(race["race_date"][0])
            month_key = race_date_str[:7]

            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            model_probs: dict[str, list[float]] = {}

            # LambdaRank
            lr_scores = models["lambdarank"].predict(X)
            lr_probs_raw = compute_all_ticket_probs(lr_scores, temperature)
            model_probs["lambdarank"] = [lr_probs_raw["win"].get(i, 0.0) for i in range(race.shape[0])]

            if is_ensemble:
                # Binary Win
                bw_raw = models["binary_win"].predict(X)
                bw_probs = expit(bw_raw).tolist()
                if "calibrator_win" in models:
                    bw_probs = models["calibrator_win"].predict(bw_raw).tolist()
                model_probs["binary_win"] = normalize_within_race(bw_probs)

                # Binary Top3
                bt3_raw = models["binary_top3"].predict(X)
                bt3_probs = expit(bt3_raw).tolist()
                if "calibrator_top3" in models:
                    bt3_probs = models["calibrator_top3"].predict(bt3_raw).tolist()
                model_probs["binary_top3"] = normalize_within_race(bt3_probs)

            for mn in model_names:
                win_probs = model_probs[mn]
                s = stats[mn]

                if month_key not in s["monthly_roi"]:
                    s["monthly_roi"][month_key] = {"bet": 0, "payout": 0}

                best_idx = int(np.argmax(win_probs))
                pp_best = pp_list[best_idx]
                actual_payout = payout_map.get(("win", str(pp_best)), 0)
                s["top_pick_bets"] += 1
                s["monthly_roi"][month_key]["bet"] += 100
                if actual_payout > 0:
                    s["top_pick_wins"] += 1
                    s["top_pick_payout"] += actual_payout
                    s["monthly_roi"][month_key]["payout"] += actual_payout

                best_odds = confirmed_odds_list[best_idx]
                if best_odds and best_odds > 0:
                    for band_name, low, high in ODDS_BANDS:
                        if low <= best_odds < high:
                            s["top_pick_by_band"][band_name]["bets"] += 1
                            if actual_payout > 0:
                                s["top_pick_by_band"][band_name]["wins"] += 1
                                s["top_pick_by_band"][band_name]["payout"] += actual_payout
                            break

                for idx in range(race.shape[0]):
                    wp = win_probs[idx]
                    conf_odds = confirmed_odds_list[idx]
                    actual_win = 1 if finish_list[idx] == 1 else 0
                    s["brier_preds"].append(wp)
                    s["brier_actuals"].append(actual_win)

                    if conf_odds and conf_odds > 0:
                        ev = wp * conf_odds
                        mip = market_implied_prob(conf_odds)
                        overlay = (wp / mip - 1.0) if mip > 0 else 0.0

                        for band_name, low, high in ODDS_BANDS:
                            if low <= conf_odds < high:
                                s["odds_band_stats"][band_name]["model_probs"].append(wp)
                                s["odds_band_stats"][band_name]["actual_wins"].append(actual_win)
                                break

                        if ev > ev_threshold:
                            s["ev_bets"] += 1
                            pp = pp_list[idx]
                            ev_actual = payout_map.get(("win", str(pp)), 0)
                            if ev_actual > 0:
                                s["ev_wins"] += 1
                                s["ev_payout"] += ev_actual

                        for threshold in OVERLAY_THRESHOLDS:
                            if overlay >= threshold:
                                pp = pp_list[idx]
                                p = payout_map.get(("win", str(pp)), 0)
                                s["overlay_bets"][threshold].append({"payout": p, "win": actual_win})

    results = {}
    for mn in model_names:
        s = stats[mn]
        tp_roi = s["top_pick_payout"] / (s["top_pick_bets"] * 100) * 100 if s["top_pick_bets"] else 0
        ev_roi = s["ev_payout"] / (s["ev_bets"] * 100) * 100 if s["ev_bets"] else 0
        brier = np.mean([(p - a) ** 2 for p, a in zip(s["brier_preds"], s["brier_actuals"])]) if s["brier_preds"] else 0

        monthly_rois = []
        for m, data in sorted(s["monthly_roi"].items()):
            if data["bet"] > 0:
                monthly_rois.append(data["payout"] / data["bet"] * 100)
        sharpe = (
            (np.mean(monthly_rois) - 100) / np.std(monthly_rois)
            if len(monthly_rois) > 1 and np.std(monthly_rois) > 0
            else 0
        )

        h1_payout = sum(d["payout"] for m, d in s["monthly_roi"].items() if m < "2024-07")
        h1_bet = sum(d["bet"] for m, d in s["monthly_roi"].items() if m < "2024-07")
        h2_payout = sum(d["payout"] for m, d in s["monthly_roi"].items() if m >= "2024-07")
        h2_bet = sum(d["bet"] for m, d in s["monthly_roi"].items() if m >= "2024-07")

        overlay_results = {}
        for t in OVERLAY_THRESHOLDS:
            bets = s["overlay_bets"][t]
            n = len(bets)
            total_payout = sum(b["payout"] for b in bets)
            wins = sum(b["win"] for b in bets)
            roi = total_payout / (n * 100) * 100 if n > 0 else 0
            overlay_results[t] = {"bets": n, "wins": wins, "roi": round(roi, 1)}

        band_results = {}
        for band_name, _, _ in ODDS_BANDS:
            bs = s["odds_band_stats"][band_name]
            mp = bs["model_probs"]
            aw = bs["actual_wins"]
            n = len(mp)
            if n > 0:
                avg_model = np.mean(mp)
                actual_rate = np.mean(aw)
                efficiency = (actual_rate / avg_model * 100) if avg_model > 0 else 0
            else:
                avg_model = actual_rate = efficiency = 0
            band_results[band_name] = {
                "count": n,
                "avg_model_prob": round(float(avg_model) * 100, 2),
                "actual_win_rate": round(float(actual_rate) * 100, 2),
                "model_efficiency": round(float(efficiency), 1),
            }

        tp_band_results = {}
        for band_name, _, _ in ODDS_BANDS:
            d = s["top_pick_by_band"][band_name]
            roi = d["payout"] / (d["bets"] * 100) * 100 if d["bets"] > 0 else 0
            hit = d["wins"] / d["bets"] * 100 if d["bets"] > 0 else 0
            tp_band_results[band_name] = {
                "bets": d["bets"], "wins": d["wins"],
                "roi": round(roi, 1), "hit_rate": round(hit, 1),
            }

        results[mn] = {
            "model_name": mn,
            "surface": surface,
            "top_pick_roi": round(tp_roi, 1),
            "top_pick_wins": s["top_pick_wins"],
            "top_pick_bets": s["top_pick_bets"],
            "top_pick_hit_rate": round(s["top_pick_wins"] / s["top_pick_bets"] * 100, 1) if s["top_pick_bets"] else 0,
            "ev_filter_roi": round(ev_roi, 1),
            "ev_bets": s["ev_bets"],
            "ev_wins": s["ev_wins"],
            "ev_hit_rate": round(s["ev_wins"] / s["ev_bets"] * 100, 1) if s["ev_bets"] else 0,
            "brier_score": round(float(brier), 4),
            "sharpe_ratio": round(float(sharpe), 2),
            "h1_roi": round(h1_payout / h1_bet * 100, 1) if h1_bet else 0,
            "h2_roi": round(h2_payout / h2_bet * 100, 1) if h2_bet else 0,
            "feature_count": len(feature_columns),
            "overlay_analysis": overlay_results,
            "odds_band_model_accuracy": band_results,
            "top_pick_by_odds_band": tp_band_results,
        }

    return results


def print_model_results(result: dict, ev_threshold: float) -> None:
    mn = result["model_name"]
    surface = result["surface"]
    print(f"\n  --- {mn.upper()} ---")

    print(f"  Brier Score:      {result['brier_score']}")
    print(f"  Sharpe Ratio:     {result['sharpe_ratio']}")
    print(f"  H1/H2 ROI:       {result['h1_roi']}% / {result['h2_roi']}%")

    print(f"  Top Pick ROI:     {result['top_pick_roi']}% ({result['top_pick_wins']}/{result['top_pick_bets']} = {result['top_pick_hit_rate']}%)")
    print(f"  EV>{ev_threshold} ROI:      {result['ev_filter_roi']}% ({result['ev_wins']}/{result['ev_bets']})")

    print(f"\n  [オーバーレイ分析]")
    print(f"  {'閾値':>8s}  {'Bets':>6s}  {'Wins':>5s}  {'ROI':>8s}")
    for t in OVERLAY_THRESHOLDS:
        d = result["overlay_analysis"][t]
        print(f"  {f'>={t:.0%}':>8s}  {d['bets']:>6d}  {d['wins']:>5d}  {d['roi']:>7.1f}%")

    print(f"\n  [モデル精度 by オッズ帯]")
    print(f"  {'帯':>8s}  {'N':>7s}  {'Model%':>8s}  {'Actual%':>8s}  {'Eff':>8s}")
    for band_name, _, _ in ODDS_BANDS:
        d = result["odds_band_model_accuracy"][band_name]
        print(f"  {band_name:>8s}  {d['count']:>7d}  {d['avg_model_prob']:>7.2f}%  {d['actual_win_rate']:>7.2f}%  {d['model_efficiency']:>7.1f}%")

    print(f"\n  [Top Pick by オッズ帯]")
    print(f"  {'帯':>8s}  {'Bets':>6s}  {'Wins':>5s}  {'Hit%':>6s}  {'ROI':>8s}")
    for band_name, _, _ in ODDS_BANDS:
        d = result["top_pick_by_odds_band"][band_name]
        if d["bets"] > 0:
            print(f"  {band_name:>8s}  {d['bets']:>6d}  {d['wins']:>5d}  {d['hit_rate']:>5.1f}%  {d['roi']:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", choices=["turf", "dirt", "both"], default="both")
    parser.add_argument("--ev-threshold", type=float, default=1.2)
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

        print(f"\n[{surface_name.upper()}] Loading all data for feature history...")
        t0 = time.time()
        df = loader.load_race_dataset(end_date=eval_end)
        surface_df = df.filter(pl.col("surface_code") == surface_code)
        print(f"[{surface_name.upper()}] Building features ({surface_df.shape[0]} rows)...")
        features = pipeline.build_features(surface_df)
        features = features.filter(
            (pl.col("race_date") >= eval_start) & (pl.col("race_date") <= eval_end)
        )
        print(f"[{surface_name.upper()}] Features ready in {time.time() - t0:.0f}s → {features.shape[0]} eval rows")

        results = evaluate_surface(surface_name, features, engine, ev_threshold=args.ev_threshold)
        print(f"\n{'=' * 60}")
        print(f"  {surface_name.upper()} — Phase 3c (confirmed odds, full history)")
        print(f"  Features: {results[list(results.keys())[0]]['feature_count']}")
        print(f"{'=' * 60}")
        for mn, result in results.items():
            print_model_results(result, args.ev_threshold)


if __name__ == "__main__":
    main()
