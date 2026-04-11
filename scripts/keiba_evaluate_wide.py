#!/usr/bin/env python3
"""Evaluate Wide (ワイド) and Quinella (馬連) strategies.

For each model, picks the top-2 horses per race and checks against
actual confirmed payouts in keiba_ticket_payouts.

Usage:
    uv run python scripts/keiba_evaluate_wide.py --surface turf
    uv run python scripts/keiba_evaluate_wide.py --surface both
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


def load_phase3c_models(surface: str) -> dict:
    base = Path(f"data/keiba/models/{surface}")
    version = (base / "latest").read_text().strip()
    version_dir = base / version
    metadata = json.loads((version_dir / "metadata.json").read_text())
    result = {"metadata": metadata, "feature_columns": metadata["feature_columns"]}

    if metadata.get("model_type") == "phase3c_ensemble":
        result["lambdarank"] = lgb.Booster(model_file=str(version_dir / "lambdarank.txt"))
        result["binary_win"] = lgb.Booster(model_file=str(version_dir / "binary_win.txt"))
        result["binary_top3"] = lgb.Booster(model_file=str(version_dir / "binary_top3.txt"))
        huber_path = version_dir / "huber.txt"
        if huber_path.exists():
            result["huber"] = lgb.Booster(model_file=str(huber_path))
        for cal_name in ("calibrator_win", "calibrator_top3"):
            cal_path = version_dir / f"{cal_name.replace('calibrator_', 'calibrator_')}.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    result[cal_name] = pickle.load(f)
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


def normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    n = len(probs)
    return [1.0 / n] * n if n > 0 else probs


def make_wide_key(pp1: int, pp2: int) -> str:
    a, b = sorted([pp1, pp2])
    return f"{a:02d}{b:02d}"


def evaluate_wide(
    surface: str,
    features: pl.DataFrame,
    engine,
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
        if "huber" in models:
            model_names.append("huber")
        model_names.extend(["ensemble_avg", "ensemble_agree"])

    stats: dict[str, dict] = {}
    for mn in model_names:
        stats[mn] = {
            "wide_bets": 0, "wide_hits": 0, "wide_payout": 0,
            "quinella_bets": 0, "quinella_hits": 0, "quinella_payout": 0,
            "monthly": {},
            "top2_both_in_top3": 0,
            "wide_by_confidence": {g: {"bets": 0, "hits": 0, "payout": 0} for g in ["low", "mid", "high"]},
            "wide_by_odds": {},
        }

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            n = race.shape[0]
            if n < 3:
                continue

            X = build_X(race, feature_columns, cat_cols)
            pp_list = race["post_position"].to_list()
            finish_list = race["finish_position"].to_list()
            confirmed_odds_list = race["confirmed_win_odds"].to_list()
            race_date_str = str(race["race_date"][0])
            month_key = race_date_str[:7]

            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            model_scores: dict[str, np.ndarray] = {}

            lr_scores = models["lambdarank"].predict(X)
            model_scores["lambdarank"] = lr_scores

            if is_ensemble:
                bw_raw = models["binary_win"].predict(X)
                if "calibrator_win" in models:
                    bw_probs = np.array(models["calibrator_win"].predict(bw_raw))
                else:
                    bw_probs = expit(bw_raw)
                model_scores["binary_win"] = bw_probs

                bt3_raw = models["binary_top3"].predict(X)
                if "calibrator_top3" in models:
                    bt3_probs = np.array(models["calibrator_top3"].predict(bt3_raw))
                else:
                    bt3_probs = expit(bt3_raw)
                model_scores["binary_top3"] = bt3_probs

                if "huber" in models:
                    huber_scores = models["huber"].predict(X)
                    model_scores["huber"] = huber_scores

                lr_norm = (lr_scores - lr_scores.min()) / (lr_scores.max() - lr_scores.min() + 1e-12)
                bw_norm = bw_probs / (bw_probs.sum() + 1e-12)
                bt3_norm = bt3_probs / (bt3_probs.sum() + 1e-12)

                if "huber" in models:
                    h_norm = (huber_scores - huber_scores.min()) / (huber_scores.max() - huber_scores.min() + 1e-12)
                    model_scores["ensemble_avg"] = 0.3 * lr_norm + 0.1 * bw_norm + 0.4 * bt3_norm + 0.2 * h_norm
                else:
                    model_scores["ensemble_avg"] = 0.3 * lr_norm + 0.3 * bw_norm + 0.4 * bt3_norm

                top2_sets = [set(np.argsort(-lr_scores)[:2]), set(np.argsort(-bt3_probs)[:2])]
                if "huber" in models:
                    top2_sets.append(set(np.argsort(-huber_scores)[:2]))
                agree_common = top2_sets[0]
                for s in top2_sets[1:]:
                    agree_common = agree_common & s
                model_scores["ensemble_agree"] = model_scores["ensemble_avg"]

            actual_top3 = {i for i, fp in enumerate(finish_list) if fp is not None and 1 <= fp <= 3}

            for mn in model_names:
                scores = model_scores[mn]
                s = stats[mn]

                if mn == "ensemble_agree" and len(agree_common) < 2:
                    continue

                top2_idx = np.argsort(-scores)[:2].tolist()
                pp1, pp2 = pp_list[top2_idx[0]], pp_list[top2_idx[1]]
                wide_key = make_wide_key(pp1, pp2)
                quinella_key = make_wide_key(pp1, pp2)

                if month_key not in s["monthly"]:
                    s["monthly"][month_key] = {"wide_bet": 0, "wide_payout": 0}

                # Wide: top-2 both in top 3
                s["wide_bets"] += 1
                s["monthly"][month_key]["wide_bet"] += 100
                wide_payout = payout_map.get(("wide", wide_key), 0)
                if wide_payout > 0:
                    s["wide_hits"] += 1
                    s["wide_payout"] += wide_payout
                    s["monthly"][month_key]["wide_payout"] += wide_payout

                if top2_idx[0] in actual_top3 and top2_idx[1] in actual_top3:
                    s["top2_both_in_top3"] += 1

                # Quinella: top-2 finish 1st and 2nd
                s["quinella_bets"] += 1
                q_payout = payout_map.get(("quinella", quinella_key), 0)
                if q_payout > 0:
                    s["quinella_hits"] += 1
                    s["quinella_payout"] += q_payout

                # Confidence: gap between 2nd and 3rd
                if n > 2:
                    sorted_scores = np.sort(scores)[::-1]
                    gap = sorted_scores[1] - sorted_scores[2]
                    total_range = sorted_scores[0] - sorted_scores[-1] if sorted_scores[0] != sorted_scores[-1] else 1.0
                    norm_gap = gap / total_range
                    if norm_gap < 0.1:
                        conf = "low"
                    elif norm_gap < 0.25:
                        conf = "mid"
                    else:
                        conf = "high"
                    s["wide_by_confidence"][conf]["bets"] += 1
                    if wide_payout > 0:
                        s["wide_by_confidence"][conf]["hits"] += 1
                        s["wide_by_confidence"][conf]["payout"] += wide_payout

                # Odds band of top pick
                top_odds = confirmed_odds_list[top2_idx[0]]
                if top_odds and top_odds > 0:
                    if top_odds < 3:
                        band = "1-3x"
                    elif top_odds < 5:
                        band = "3-5x"
                    elif top_odds < 10:
                        band = "5-10x"
                    elif top_odds < 20:
                        band = "10-20x"
                    else:
                        band = "20x+"
                    if band not in s["wide_by_odds"]:
                        s["wide_by_odds"][band] = {"bets": 0, "hits": 0, "payout": 0}
                    s["wide_by_odds"][band]["bets"] += 1
                    if wide_payout > 0:
                        s["wide_by_odds"][band]["hits"] += 1
                        s["wide_by_odds"][band]["payout"] += wide_payout

    results = {}
    for mn in model_names:
        s = stats[mn]
        wb, wh, wp = s["wide_bets"], s["wide_hits"], s["wide_payout"]
        qb, qh, qp = s["quinella_bets"], s["quinella_hits"], s["quinella_payout"]

        w_roi = wp / (wb * 100) * 100 if wb else 0
        q_roi = qp / (qb * 100) * 100 if qb else 0
        w_hit = wh / wb * 100 if wb else 0
        q_hit = qh / qb * 100 if qb else 0

        monthly_rois = []
        for m, d in sorted(s["monthly"].items()):
            if d["wide_bet"] > 0:
                monthly_rois.append(d["wide_payout"] / d["wide_bet"] * 100)
        sharpe = (
            (np.mean(monthly_rois) - 100) / np.std(monthly_rois)
            if len(monthly_rois) > 1 and np.std(monthly_rois) > 0 else 0
        )

        results[mn] = {
            "model": mn,
            "wide_roi": round(w_roi, 1),
            "wide_hits": wh, "wide_bets": wb,
            "wide_hit_rate": round(w_hit, 1),
            "wide_avg_payout": round(wp / wh) if wh else 0,
            "quinella_roi": round(q_roi, 1),
            "quinella_hits": qh, "quinella_bets": qb,
            "quinella_hit_rate": round(q_hit, 1),
            "quinella_avg_payout": round(qp / qh) if qh else 0,
            "top2_in_top3_rate": round(s["top2_both_in_top3"] / wb * 100, 1) if wb else 0,
            "sharpe": round(float(sharpe), 2),
            "by_confidence": {
                k: {
                    "bets": v["bets"],
                    "hits": v["hits"],
                    "roi": round(v["payout"] / (v["bets"] * 100) * 100, 1) if v["bets"] else 0,
                    "hit_rate": round(v["hits"] / v["bets"] * 100, 1) if v["bets"] else 0,
                } for k, v in s["wide_by_confidence"].items()
            },
            "by_odds": {
                k: {
                    "bets": v["bets"],
                    "hits": v["hits"],
                    "roi": round(v["payout"] / (v["bets"] * 100) * 100, 1) if v["bets"] else 0,
                    "hit_rate": round(v["hits"] / v["bets"] * 100, 1) if v["bets"] else 0,
                } for k, v in s["wide_by_odds"].items()
            },
        }

    return results


def print_results(surface: str, results: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {surface.upper()} — Wide/Quinella Strategy Evaluation (confirmed payouts)")
    print(f"{'=' * 70}")

    for mn, r in results.items():
        print(f"\n  --- {mn.upper()} ---")
        print(f"  ワイド◎-○: ROI {r['wide_roi']}% | {r['wide_hits']}/{r['wide_bets']} ({r['wide_hit_rate']}%) | avg ¥{r['wide_avg_payout']}")
        print(f"  馬連◎-○:   ROI {r['quinella_roi']}% | {r['quinella_hits']}/{r['quinella_bets']} ({r['quinella_hit_rate']}%) | avg ¥{r['quinella_avg_payout']}")
        print(f"  Top2 in Top3: {r['top2_in_top3_rate']}% | Sharpe: {r['sharpe']}")

        print(f"\n  [ワイド by 確信度 (2位-3位スコアギャップ)]")
        print(f"  {'Level':>6s}  {'Bets':>6s}  {'Hit%':>6s}  {'ROI':>8s}")
        for k in ["low", "mid", "high"]:
            d = r["by_confidence"][k]
            if d["bets"] > 0:
                print(f"  {k:>6s}  {d['bets']:>6d}  {d['hit_rate']:>5.1f}%  {d['roi']:>7.1f}%")

        print(f"\n  [ワイド by ◎の確定オッズ帯]")
        print(f"  {'Band':>6s}  {'Bets':>6s}  {'Hit%':>6s}  {'ROI':>8s}")
        for k in ["1-3x", "3-5x", "5-10x", "10-20x", "20x+"]:
            if k in r["by_odds"]:
                d = r["by_odds"][k]
                if d["bets"] > 0:
                    print(f"  {k:>6s}  {d['bets']:>6d}  {d['hit_rate']:>5.1f}%  {d['roi']:>7.1f}%")


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

        results = evaluate_wide(surface_name, features, engine)
        print_results(surface_name, results)


if __name__ == "__main__":
    main()
