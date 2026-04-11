"""Training CLI command."""

from __future__ import annotations

from datetime import date, time as dt_time
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def train_command(
    train_end: str | None = typer.Option(None, help="Manual train end date (YYYY-MM-DD)"),
    val_end: str | None = typer.Option(None, help="Manual validation end date (YYYY-MM-DD)"),
    optimize: bool = typer.Option(False, help="Run Optuna hyperparameter optimization"),
    n_trials: int = typer.Option(30, help="Number of Optuna trials"),
    rebuild_features: bool = typer.Option(False, help="Force rebuild feature cache"),
    compare_with: str | None = typer.Option(None, help="Compare against an existing model version"),
    shap_samples: int = typer.Option(1000, help="Sample size for SHAP importance"),
    description: str | None = typer.Option(None, help="Human-readable description of this training run"),
    backtest: bool = typer.Option(True, "--backtest/--no-backtest", help="Run standardized backtest on test set"),
    ensemble: bool = typer.Option(False, help="Train 4-model ensemble (LambdaRank + BinaryTop2 + BinaryWin + Huber)"),
) -> None:
    import lightgbm as lgb
    import numpy as np
    import sklearn

    from providence.features.loader import DataLoader
    from providence.features.pipeline import FeaturePipeline
    from providence.model.evaluator import Evaluator
    from providence.model.split import SplitStrategy, apply_split
    from providence.model.store import ModelStore
    from providence.model.trainer import Trainer
    from providence.probability.calibration import TemperatureScaler

    if (train_end and not val_end) or (val_end and not train_end):
        console.print("[red]--train-end と --val-end は両方指定するか、両方省略してください。[/red]")
        raise typer.Exit(1)

    loader = DataLoader()
    pipeline = FeaturePipeline()
    trainer = Trainer(pipeline=pipeline)
    evaluator = Evaluator(pipeline=pipeline)
    store = ModelStore()

    raw_df = loader.load_all()
    cache_key = FeaturePipeline.cache_key(
        {
            "rows": len(raw_df),
            "race_min": raw_df["race_id"].min(),
            "race_max": raw_df["race_id"].max(),
            "entry_max": raw_df["race_entry_id"].max(),
            "date_min": raw_df["race_date"].min(),
            "date_max": raw_df["race_date"].max(),
        }
    )
    cache_path = f"data/processed/features_{cache_key}.parquet"
    if rebuild_features:
        pipeline.invalidate_cache()
    features = pipeline.build_and_cache(raw_df, cache_path)

    splitter = SplitStrategy()
    if train_end and val_end:
        split = splitter.manual_split(features, date.fromisoformat(train_end), date.fromisoformat(val_end))
    else:
        split = splitter.auto_split(features)
    splits = apply_split(features, split)

    if ensemble:
        version, version_dir, metrics, gate, compare_result, metadata = _train_ensemble(
            trainer=trainer,
            evaluator=evaluator,
            store=store,
            splits=splits,
            split=split,
            features=features,
            raw_df=raw_df,
            cache_path=cache_path,
            description=description,
            compare_with=compare_with,
            shap_samples=shap_samples,
        )
    else:
        version, version_dir, metrics, gate, compare_result, metadata = _train_single(
            trainer=trainer,
            evaluator=evaluator,
            store=store,
            splits=splits,
            split=split,
            features=features,
            raw_df=raw_df,
            cache_path=cache_path,
            n_trials=n_trials,
            optimize=optimize,
            description=description,
            compare_with=compare_with,
            shap_samples=shap_samples,
        )

    console.print(f"[green]Saved model {version}[/green]")
    console.print(metrics)
    console.print(gate)
    if compare_result:
        console.print(compare_result)

    if backtest:
        console.print("\n[bold]Standardized backtest on test set...[/bold]")
        backtest_summary = _run_standardized_backtest(
            version=version,
            test_start=split.test_start,
            test_end=split.test_end,
        )
        if backtest_summary:
            metadata["backtest"] = backtest_summary
            _save_metadata(version_dir / "metadata.json", metadata, version)
            console.print(
                f"  ROI: {backtest_summary['roi'] * 100:+.1f}%  "
                f"profit: {backtest_summary['total_profit']:+,.0f}円  "
                f"hit_rate: {backtest_summary['hit_rate'] * 100:.1f}%  "
                f"races: {backtest_summary['profit_evaluated_races']}"
            )
        else:
            console.print("[yellow]  バックテストデータ不足（オッズ/払戻未取得）。後から実行可能。[/yellow]")


def _train_single(
    *,
    trainer,
    evaluator,
    store,
    splits,
    split,
    features,
    raw_df,
    cache_path,
    n_trials,
    optimize,
    description,
    compare_with,
    shap_samples,
):
    import lightgbm as lgb
    import numpy as np
    import sklearn

    from providence.probability.calibration import IsotonicCalibrator, TemperatureScaler

    params = trainer.optimize_hyperparams(splits["train"], splits["val"], n_trials=n_trials) if optimize else None
    artifacts = trainer.train_lambdarank(splits["train"], splits["val"], params=params)

    val_scores = _scores_per_race(artifacts.model, splits["val"], artifacts.feature_columns)
    if not val_scores:
        console.print("[red]validation split から winner を持つレースを抽出できませんでした。[/red]")
        raise typer.Exit(1)
    scaler = TemperatureScaler()
    scaler.fit(
        [scores for _, scores, _ in val_scores],
        [winner for _, _, winner in val_scores],
        n_trials=max(10, min(n_trials, 50)),
    )

    console.print("  Fitting isotonic calibrator...")
    isotonic = IsotonicCalibrator()
    isotonic.fit(
        [scores for _, scores, _ in val_scores],
        [winner for _, _, winner in val_scores],
        temperature=scaler.temperature,
    )

    metrics = evaluator.evaluate(artifacts.model, splits["test"], temperature=scaler.temperature)
    importance = evaluator.feature_importance(artifacts.model)
    feature_stats = evaluator.feature_stats(features, artifacts.feature_columns)
    shap_importance = evaluator.shap_analysis(artifacts.model, splits["test"], n_samples=shap_samples)
    gate = _gate_result(metrics)
    compare_result = _compare_with_existing(store, compare_with, metrics) if compare_with else None
    metadata = {
        "model_type": artifacts.model_type,
        "params": artifacts.best_params,
        "temperature": scaler.temperature,
        "metrics": metrics,
        "gate": gate,
        "compare_with": compare_result,
        "feature_columns": artifacts.feature_columns,
        "random_seed": trainer.random_seed,
        "description": description,
        "git_hash": _get_git_hash(),
        "library_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "split": _split_dict(split),
        "trained_through_date": split.val_end.isoformat(),
        "validation_end_date": split.val_end.isoformat(),
        "data_range": {
            "start": raw_df["race_date"].min().isoformat(),
            "end": raw_df["race_date"].max().isoformat(),
        },
        "feature_cache": str(Path(cache_path).name),
    }
    metadata["calibration"] = "isotonic"
    version = store.save(artifacts.model, metadata)
    version_dir = store.version_dir(version)
    isotonic.save(version_dir / "isotonic_calibrator.pkl")
    importance.write_csv(version_dir / "feature_importance.csv")
    (version_dir / "feature_stats.json").write_text(_json_dump(evaluator.feature_stats(features, artifacts.feature_columns)))
    shap_importance.write_csv(version_dir / "shap_importance.csv")
    return version, version_dir, metrics, gate, compare_result, metadata


def _train_ensemble(
    *,
    trainer,
    evaluator,
    store,
    splits,
    split,
    features,
    raw_df,
    cache_path,
    description,
    compare_with,
    shap_samples,
):
    import lightgbm as lgb
    import numpy as np
    import sklearn

    from providence.model.ensemble import DEFAULT_WEIGHTS, combine_race_scores
    from providence.probability.calibration import IsotonicCalibrator
    from providence.probability.plackett_luce import compute_all_ticket_probs_from_strengths

    console.print("[bold]Training 6-model ensemble (V012)...[/bold]")

    console.print("  [1/6] LambdaRank...")
    rank_art = trainer.train_lambdarank(splits["train"], splits["val"])
    console.print("  [2/6] XE-NDCG...")
    xendcg_art = trainer.train_xendcg(splits["train"], splits["val"])
    console.print("  [3/6] Binary top-2...")
    top2_art = trainer.train_binary_top2(splits["train"], splits["val"])
    console.print("  [4/6] Focal Value (top-2)...")
    focal_art = trainer.train_focal_value(splits["train"], splits["val"])
    console.print("  [5/6] Binary win...")
    win_art = trainer.train_binary_win(splits["train"], splits["val"])
    console.print("  [6/6] Huber regression...")
    huber_art = trainer.train_huber(splits["train"], splits["val"])

    models = {
        "lambdarank": rank_art.model,
        "xendcg": xendcg_art.model,
        "binary_top2": top2_art.model,
        "focal_value": focal_art.model,
        "binary_win": win_art.model,
        "huber": huber_art.model,
    }
    weights = dict(DEFAULT_WEIGHTS)

    console.print("  Fitting isotonic calibrator on ensemble...")
    val_scores, val_top2_probs, val_exacta_targets = _ensemble_scores_per_race(
        models, splits["val"], rank_art.feature_columns, trainer.pipeline
    )
    isotonic = IsotonicCalibrator()
    if val_scores:
        isotonic.fit(
            [scores for _, scores, _ in val_scores],
            [winner for _, _, winner in val_scores],
            temperature=1.0,
        )

    pair_blend_alpha = _choose_pair_blend_alpha(val_scores, val_top2_probs, val_exacta_targets, isotonic)
    metrics = _evaluate_ensemble(models, splits["test"], rank_art.feature_columns, trainer.pipeline, isotonic, pair_blend_alpha)
    importance = evaluator.feature_importance(rank_art.model)
    shap_importance = evaluator.shap_analysis(rank_art.model, splits["test"], n_samples=shap_samples)
    gate = _gate_result(metrics)
    compare_result = _compare_with_existing(store, compare_with, metrics) if compare_with else None

    metadata = {
        "model_type": "ensemble",
        "ensemble_keys": list(models.keys()),
        "ensemble_weights": weights,
        "params": {
            "lambdarank": rank_art.best_params,
            "xendcg": xendcg_art.best_params,
            "binary_top2": top2_art.best_params,
            "focal_value": focal_art.best_params,
            "binary_win": win_art.best_params,
            "huber": huber_art.best_params,
        },
        "temperature": 1.0,
        "metrics": metrics,
        "gate": gate,
        "compare_with": compare_result,
        "feature_columns": rank_art.feature_columns,
        "calibration_input": "ensemble_combined",
        "pair_blend_alpha": pair_blend_alpha,
        "random_seed": trainer.random_seed,
        "description": description,
        "git_hash": _get_git_hash(),
        "library_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "split": _split_dict(split),
        "trained_through_date": split.val_end.isoformat(),
        "validation_end_date": split.val_end.isoformat(),
        "data_range": {
            "start": raw_df["race_date"].min().isoformat(),
            "end": raw_df["race_date"].max().isoformat(),
        },
        "feature_cache": str(Path(cache_path).name),
    }

    metadata["calibration"] = "isotonic" if isotonic.fitted else "none"
    version = store.save_ensemble(models, weights, metadata)
    version_dir = store.version_dir(version)
    if isotonic.fitted:
        isotonic.save(version_dir / "isotonic_calibrator.pkl")
    importance.write_csv(version_dir / "feature_importance.csv")
    (version_dir / "feature_stats.json").write_text(_json_dump(evaluator.feature_stats(features, rank_art.feature_columns)))
    shap_importance.write_csv(version_dir / "shap_importance.csv")

    console.print(f"  Ensemble weights: {weights}")
    console.print(f"  Pair blend alpha: {pair_blend_alpha}")
    return version, version_dir, metrics, gate, compare_result, metadata


def _split_dict(split) -> dict:
    return {
        "warmup": [split.warmup_start.isoformat(), split.warmup_end.isoformat()],
        "train": [split.train_start.isoformat(), split.train_end.isoformat()],
        "val": [split.val_start.isoformat(), split.val_end.isoformat()],
        "test": [split.test_start.isoformat(), split.test_end.isoformat()],
    }


def _ensemble_scores_per_race(models, df, feature_columns, pipeline):
    import numpy as np
    import pandas as pd
    import polars as pl

    from providence.features.pipeline import FeaturePipeline
    from providence.model.evaluator import _winner_index
    from providence.model.ensemble import combine_race_scores
    from providence.model.predictor import _build_X

    ordered = df.sort(["race_date", "race_number", "race_id", "post_position"])
    cat_cols = getattr(pipeline, "categorical_columns", FeaturePipeline.categorical_columns)
    groups = ordered.partition_by("race_id", maintain_order=True)
    score_rows = []
    top2_rows = []
    exacta_targets = []

    for group in groups:
        X = _build_X(group, feature_columns, categorical_columns=cat_cols)
        raw_outputs = {key: np.asarray(model.predict(X), dtype=float) for key, model in models.items()}
        scores = combine_race_scores(raw_outputs)
        finish_positions = group["finish_position"].to_list()
        winner_idx = _winner_index(finish_positions)
        if winner_idx is None:
            continue
        runner_up_idx = None
        for idx, pos in enumerate(finish_positions):
            if pos == 2:
                runner_up_idx = idx
                break
        score_rows.append((group["race_id"][0], scores, winner_idx))
        top2 = raw_outputs.get("binary_top2")
        if top2 is not None:
            top2 = np.clip(np.asarray(top2, dtype=float), 1e-8, 1.0 - 1e-8)
            top2 = top2 / top2.sum() if top2.sum() > 0 else top2
        top2_rows.append(top2)
        exacta_targets.append((winner_idx, runner_up_idx))
    return score_rows, top2_rows, exacta_targets


def _choose_pair_blend_alpha(score_rows, top2_rows, exacta_targets, isotonic) -> float:
    import numpy as np

    from providence.probability.plackett_luce import compute_all_ticket_probs_from_strengths, compute_win_probs

    alphas = [0.0, 0.1, 0.2, 0.3]
    if not score_rows or not top2_rows:
        return 0.0

    best_alpha = 0.0
    best_loss = float("inf")
    for alpha in alphas:
        losses = []
        for (race_id, scores, winner_idx), top2, target in zip(score_rows, top2_rows, exacta_targets, strict=False):
            if top2 is None:
                continue
            if target[1] is None:
                continue
            raw_win_probs = compute_win_probs(scores, 1.0)
            calibrated = isotonic.calibrate(raw_win_probs) if isotonic.fitted else raw_win_probs
            main_probs = compute_all_ticket_probs_from_strengths(calibrated)
            aux_probs = compute_all_ticket_probs_from_strengths(top2)
            # only exacta log loss on actual 1-2 order
            exacta = {
                combo: (1.0 - alpha) * prob + alpha * aux_probs["exacta"][combo]
                for combo, prob in main_probs["exacta"].items()
            }
            target_combo = target
            p = max(float(exacta[target_combo]), 1e-12)
            losses.append(-np.log(p))
        if losses:
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
    return best_alpha


def _evaluate_ensemble(models, test_df, feature_columns, pipeline, isotonic, pair_blend_alpha):
    import numpy as np
    import polars as pl
    from sklearn.metrics import brier_score_loss

    from providence.features.pipeline import FeaturePipeline
    from providence.model.ensemble import combine_race_scores
    from providence.model.evaluator import _uniform_brier_baseline, _winner_index
    from providence.model.predictor import _blend_pair_ticket_probs, _build_X
    from providence.probability.plackett_luce import compute_all_ticket_probs_from_strengths, compute_win_probs

    ordered = test_df.sort(["race_date", "race_number", "race_id", "post_position"])
    cat_cols = getattr(pipeline, "categorical_columns", FeaturePipeline.categorical_columns)
    groups = ordered.partition_by("race_id", maintain_order=True)

    win_hits = 0
    top3_overlap_total = 0.0
    win_prob_preds = []
    win_labels = []
    evaluated_race_count = 0

    for group in groups:
        X = _build_X(group, feature_columns, categorical_columns=cat_cols)
        raw_outputs = {key: np.asarray(model.predict(X), dtype=float) for key, model in models.items()}
        scores = combine_race_scores(raw_outputs)
        finish_positions = group["finish_position"].to_list()
        winner_idx = _winner_index(finish_positions)
        if winner_idx is None:
            continue
        evaluated_race_count += 1

        raw_win_probs = compute_win_probs(scores, 1.0)
        calibrated = isotonic.calibrate(raw_win_probs) if isotonic.fitted else raw_win_probs
        ticket_probs = compute_all_ticket_probs_from_strengths(calibrated)
        top2 = raw_outputs.get("binary_top2")
        if top2 is not None:
            top2 = np.clip(np.asarray(top2, dtype=float), 1e-8, 1.0 - 1e-8)
            top2 = top2 / top2.sum() if top2.sum() > 0 else top2
        ticket_probs = _blend_pair_ticket_probs(ticket_probs, top2, pair_blend_alpha)

        predicted_order = np.argsort(-np.asarray(list(ticket_probs["win"].values()), dtype=float))
        if predicted_order[0] == winner_idx:
            win_hits += 1
        actual_top3 = {idx for idx, pos in enumerate(finish_positions) if pos is not None and 1 <= pos <= 3}
        predicted_top3 = set(predicted_order[: min(3, len(predicted_order))])
        top3_overlap_total += len(actual_top3 & predicted_top3) / 3.0
        for idx, prob in ticket_probs["win"].items():
            win_prob_preds.append(float(prob))
            win_labels.append(1 if idx == winner_idx else 0)

    baseline = _uniform_brier_baseline(groups)
    denom = evaluated_race_count or 1
    return {
        "win_accuracy": win_hits / denom,
        "top3_overlap": top3_overlap_total / denom,
        "brier_score": brier_score_loss(win_labels, win_prob_preds) if win_labels else float("nan"),
        "brier_baseline": baseline,
    }


def _get_git_hash() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _run_standardized_backtest(
    *,
    version: str,
    test_start: date,
    test_end: date,
) -> dict | None:
    """Run a standardized backtest on the test period and return summary dict."""
    from providence.backtest.engine import BacktestEngine
    from providence.backtest.metrics import summarize_backtest
    from providence.strategy.types import EvaluationMode, StrategyConfig

    engine = BacktestEngine()
    config = StrategyConfig()
    try:
        results = engine.run(
            start_date=test_start,
            end_date=test_end,
            judgment_clock=dt_time(15, 0, 0),
            evaluation_mode=EvaluationMode.FIXED,
            model_version=version,
            config=config,
            use_final_odds=False,
        )
    except Exception as exc:
        console.print(f"[yellow]  バックテスト実行エラー: {exc}[/yellow]")
        return None

    if not results:
        return None

    summary = summarize_backtest(results)
    if summary.profit_evaluated_races == 0:
        return None

    bet_races = sum(1 for r in results if r.total_stake > 0)
    hit_in_bet = sum(1 for r in results if r.total_stake > 0 and any(s.hit for s in r.settled_recommendations))

    return {
        "total_races": summary.total_races,
        "profit_evaluated_races": summary.profit_evaluated_races,
        "bet_races": bet_races,
        "hit_in_bet_races": hit_in_bet,
        "total_stake": summary.total_stake,
        "total_payout": summary.total_payout,
        "total_profit": summary.total_profit,
        "roi": summary.roi,
        "hit_rate": summary.hit_rate,
        "bet_hit_rate": (hit_in_bet / bet_races) if bet_races > 0 else 0.0,
        "max_drawdown": summary.max_drawdown,
        "sharpe_ratio": summary.sharpe_ratio,
        "config": {
            "fractional_kelly": config.fractional_kelly,
            "min_expected_value": config.min_expected_value,
            "min_confidence": config.min_confidence,
            "max_candidates": config.max_candidates,
            "max_total_stake": config.max_total_stake,
        },
        "period": {
            "start": test_start.isoformat(),
            "end": test_end.isoformat(),
        },
        "use_final_odds": False,
    }


def _save_metadata(path: Path, metadata: dict, version: str) -> None:
    import json
    from datetime import UTC, datetime

    metadata["version"] = version
    metadata["created_at"] = metadata.get("created_at", datetime.now(UTC).isoformat())
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def _scores_per_race(model, df, feature_columns):
    import pandas as pd
    import polars as pl

    from providence.features.pipeline import FeaturePipeline

    ordered = df.sort(["race_date", "race_number", "race_id", "post_position"])

    X = pd.DataFrame(
        {
            column: (
                ordered[column].cast(pl.Int32).fill_null(-1).to_list()
                if column in FeaturePipeline.categorical_columns
                else ordered[column].cast(pl.Float64).fill_null(float("nan")).to_list()
            )
            for column in feature_columns
        }
    )
    scores = model.predict(X)
    out = ordered.with_columns(pl.Series("score", scores))
    groups = out.partition_by("race_id", maintain_order=True)
    result = []
    for group in groups:
        winner_idx = None
        for idx, pos in enumerate(group["finish_position"].to_list()):
            if pos == 1:
                winner_idx = idx
                break
        if winner_idx is None:
            continue
        result.append((group["race_id"][0], group["score"].to_numpy(), winner_idx))
    return result


def _gate_result(metrics: dict[str, float]) -> dict[str, object]:
    checks = {
        "win_accuracy": metrics.get("win_accuracy", 0.0) >= 0.25,
        "top3_overlap": metrics.get("top3_overlap", 0.0) >= 0.50,
        "brier_score": metrics.get("brier_score", 1.0) < metrics.get("brier_baseline", 1.0),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def _compare_with_existing(store: ModelStore, version: str, new_metrics: dict[str, float]) -> dict[str, object]:
    try:
        _, metadata = store.load(version)
    except FileNotFoundError:
        return {"version": version, "error": "not_found"}

    old_metrics = metadata.get("metrics", {})
    comparison = {}
    for key in ("win_accuracy", "top3_overlap", "brier_score"):
        if key in old_metrics and key in new_metrics:
            comparison[key] = {
                "old": old_metrics[key],
                "new": new_metrics[key],
                "delta": new_metrics[key] - old_metrics[key],
            }
    return {"version": version, "comparison": comparison}


def _json_dump(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
