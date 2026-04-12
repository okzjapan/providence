#!/usr/bin/env python3
"""Full strategy evaluation: all ticket types × EV filtering × Kelly sizing.

Applies the autorace-proven approach to keiba:
1. Compute model probabilities for ALL ticket combinations
2. Compute EV = model_prob × confirmed_odds for each
3. Filter to positive-EV bets only
4. Apply Kelly Criterion sizing
5. Evaluate ROI

Win: uses confirmed_win_odds (available for all horses)
Wide/Quinella/etc: uses confirmed payouts for evaluation

Usage:
    uv run python scripts/keiba_strategy_full.py --surface turf --year 2024
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
    vdir = base / version
    metadata = json.loads((vdir / "metadata.json").read_text())
    result = {"metadata": metadata, "feature_columns": metadata["feature_columns"]}
    for name in ("lambdarank", "binary_win", "binary_top3", "huber"):
        p = vdir / f"{name}.txt"
        if p.exists():
            result[name] = lgb.Booster(model_file=str(p))
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


def evaluate(surface: str, features: pl.DataFrame, engine, year: int) -> None:
    surface_code = 1 if surface == "turf" else 2
    surface_df = features.filter(pl.col("surface_code") == surface_code)

    models = load_models(surface)
    feat_cols = models["feature_columns"]
    temperature = models["metadata"].get("temperature", 1.0)
    pipeline = KeibaFeaturePipeline()
    cat_cols = pipeline.categorical_columns
    race_ids = surface_df["race_id"].unique().sort().to_list()

    # Strategy results: {strategy_name: {ticket_type: stats}}
    ev_thresholds = [0.0, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    ticket_types_eval = ["win", "wide", "quinella", "exacta", "trio", "trifecta"]

    results: dict[str, dict] = {}
    for ev_th in ev_thresholds:
        for tt in ticket_types_eval:
            key = f"ev>={ev_th:.2f}_{tt}"
            results[key] = {"bets": 0, "total_stake": 0, "total_payout": 0, "hits": 0}

    with engine.connect() as conn:
        for rid in race_ids:
            race = surface_df.filter(pl.col("race_id") == rid).sort("post_position")
            n = race.shape[0]
            if n < 3:
                continue

            X = build_X(race, feat_cols, cat_cols)
            pp_list = race["post_position"].to_list()
            odds_list = race["confirmed_win_odds"].to_list()

            # Model scores → Plackett-Luce probabilities
            lr_scores = models["lambdarank"].predict(X)
            all_probs = compute_all_ticket_probs(lr_scores, temperature)

            # Also get calibrated win probs from BinaryWin
            if "binary_win" in models:
                bw_raw = models["binary_win"].predict(X)
                if "calibrator_win" in models:
                    bw_probs = np.array(models["calibrator_win"].predict(bw_raw))
                else:
                    bw_probs = expit(bw_raw)
                bw_norm = bw_probs / (bw_probs.sum() + 1e-12)
            else:
                bw_norm = np.array([all_probs["win"].get(i, 0) for i in range(n)])

            # Load actual payouts
            payouts_raw = conn.execute(
                text("SELECT ticket_type, combination, payout_amount FROM keiba_ticket_payouts WHERE race_id = :rid"),
                {"rid": int(rid)},
            ).fetchall()
            payout_map = {(r[0], r[1]): r[2] for r in payouts_raw}

            # --- WIN: EV = calibrated_prob × confirmed_odds ---
            for idx in range(n):
                win_prob = float(bw_norm[idx])
                conf_odds = odds_list[idx]
                if conf_odds is None or conf_odds <= 0:
                    continue

                ev = win_prob * conf_odds
                pp = pp_list[idx]
                payout = payout_map.get(("win", str(pp)), 0)

                # Kelly fraction: (prob * odds - 1) / (odds - 1)
                kelly = max(0, (win_prob * conf_odds - 1) / (conf_odds - 1)) if conf_odds > 1 else 0
                kelly_bet = kelly * 0.25 * 1000  # fractional Kelly × bankroll unit

                for ev_th in ev_thresholds:
                    key = f"ev>={ev_th:.2f}_win"
                    if ev >= ev_th and kelly_bet >= 100:
                        stake = max(100, round(kelly_bet / 100) * 100)
                        results[key]["bets"] += 1
                        results[key]["total_stake"] += stake
                        if payout > 0:
                            results[key]["hits"] += 1
                            results[key]["total_payout"] += payout * (stake / 100)

            # --- WIDE: model_prob from Plackett-Luce ---
            wide_probs = all_probs.get("wide", {})
            for (i, j), prob in wide_probs.items():
                if i >= n or j >= n:
                    continue
                pp_i, pp_j = pp_list[i], pp_list[j]
                a, b = sorted([pp_i, pp_j])
                wide_key = f"{a:02d}{b:02d}"
                payout = payout_map.get(("wide", wide_key), 0)

                # Estimate Wide odds from individual Win odds
                odds_i = odds_list[i] if odds_list[i] and odds_list[i] > 0 else 999
                odds_j = odds_list[j] if odds_list[j] and odds_list[j] > 0 else 999
                est_wide_odds = 0.775 / max(prob, 1e-6) if prob > 0.01 else 999

                ev = prob * est_wide_odds
                kelly = max(0, (prob * est_wide_odds - 1) / (est_wide_odds - 1)) if est_wide_odds > 1 else 0
                kelly_bet = kelly * 0.25 * 1000

                for ev_th in ev_thresholds:
                    key = f"ev>={ev_th:.2f}_wide"
                    if ev >= ev_th and kelly_bet >= 100:
                        stake = max(100, round(kelly_bet / 100) * 100)
                        results[key]["bets"] += 1
                        results[key]["total_stake"] += stake
                        if payout > 0:
                            results[key]["hits"] += 1
                            results[key]["total_payout"] += payout * (stake / 100)

            # --- QUINELLA: model_prob from Plackett-Luce ---
            quin_probs = all_probs.get("quinella", {})
            for (i, j), prob in quin_probs.items():
                if i >= n or j >= n:
                    continue
                pp_i, pp_j = pp_list[i], pp_list[j]
                a, b = sorted([pp_i, pp_j])
                quin_key = f"{a:02d}{b:02d}"
                payout = payout_map.get(("quinella", quin_key), 0)

                est_odds = 0.775 / max(prob, 1e-6) if prob > 0.01 else 999
                ev = prob * est_odds
                kelly = max(0, (prob * est_odds - 1) / (est_odds - 1)) if est_odds > 1 else 0
                kelly_bet = kelly * 0.25 * 1000

                for ev_th in ev_thresholds:
                    key = f"ev>={ev_th:.2f}_quinella"
                    if ev >= ev_th and kelly_bet >= 100:
                        stake = max(100, round(kelly_bet / 100) * 100)
                        results[key]["bets"] += 1
                        results[key]["total_stake"] += stake
                        if payout > 0:
                            results[key]["hits"] += 1
                            results[key]["total_payout"] += payout * (stake / 100)

            # --- EXACTA ---
            exacta_probs = all_probs.get("exacta", {})
            for (i, j), prob in exacta_probs.items():
                if i >= n or j >= n:
                    continue
                pp_i, pp_j = pp_list[i], pp_list[j]
                exacta_key = f"{pp_i:02d}{pp_j:02d}"
                payout = payout_map.get(("exacta", exacta_key), 0)

                est_odds = 0.75 / max(prob, 1e-6) if prob > 0.005 else 999
                ev = prob * est_odds
                kelly = max(0, (prob * est_odds - 1) / (est_odds - 1)) if est_odds > 1 else 0
                kelly_bet = kelly * 0.25 * 1000

                for ev_th in ev_thresholds:
                    key = f"ev>={ev_th:.2f}_exacta"
                    if ev >= ev_th and kelly_bet >= 100:
                        stake = max(100, round(kelly_bet / 100) * 100)
                        results[key]["bets"] += 1
                        results[key]["total_stake"] += stake
                        if payout > 0:
                            results[key]["hits"] += 1
                            results[key]["total_payout"] += payout * (stake / 100)

            # --- TRIO (三連複) ---
            trio_probs = all_probs.get("trio", {})
            for combo, prob in trio_probs.items():
                if any(idx >= n for idx in combo):
                    continue
                pps = tuple(sorted([pp_list[idx] for idx in combo]))
                trio_key = f"{pps[0]:02d}{pps[1]:02d}{pps[2]:02d}"
                payout = payout_map.get(("trio", trio_key), 0)

                est_odds = 0.75 / max(prob, 1e-6) if prob > 0.003 else 999
                ev = prob * est_odds
                kelly = max(0, (prob * est_odds - 1) / (est_odds - 1)) if est_odds > 1 else 0
                kelly_bet = kelly * 0.25 * 1000

                for ev_th in ev_thresholds:
                    key = f"ev>={ev_th:.2f}_trio"
                    if ev >= ev_th and kelly_bet >= 100:
                        stake = max(100, round(kelly_bet / 100) * 100)
                        results[key]["bets"] += 1
                        results[key]["total_stake"] += stake
                        if payout > 0:
                            results[key]["hits"] += 1
                            results[key]["total_payout"] += payout * (stake / 100)

            # Skip trifecta for performance (too many combinations)

    # Print results
    print(f"\n{'=' * 85}")
    print(f"  {surface.upper()} {year} — Full Strategy Evaluation (EV filter + Kelly sizing)")
    print(f"  Model: v006, 140 features, 4 models")
    print(f"{'=' * 85}")
    print(f"  {'Strategy':<25s} {'Bets':>7s} {'Hits':>6s} {'Hit%':>6s} {'Stake':>10s} {'Payout':>10s} {'ROI':>8s}")
    print(f"  {'-' * 75}")

    for ev_th in ev_thresholds:
        for tt in ticket_types_eval:
            key = f"ev>={ev_th:.2f}_{tt}"
            r = results[key]
            if r["bets"] == 0:
                continue
            roi = r["total_payout"] / r["total_stake"] * 100 if r["total_stake"] > 0 else 0
            hit = r["hits"] / r["bets"] * 100
            marker = " ★" if roi >= 100 else " ●" if roi >= 90 else ""
            print(f"  {key:<25s} {r['bets']:>7d} {r['hits']:>6d} {hit:>5.1f}% {r['total_stake']:>9.0f} {r['total_payout']:>9.0f} {roi:>6.1f}%{marker}")
        if any(results[f"ev>={ev_th:.2f}_{tt}"]["bets"] > 0 for tt in ticket_types_eval):
            print()


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
        evaluate(sname, features, engine, args.year)


if __name__ == "__main__":
    main()
