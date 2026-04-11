"""Model B race quality filter.

Computes a combined score from Model A (ability ranking) and Model B (market
error prediction), then filters races below a calibrated quality threshold.

The threshold is pre-computed from historical data and stored in a JSON file.
To calibrate, run ``providence autobet calibrate-model-b``.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import structlog

from providence.model.value_model import predict_value_scores

log = structlog.get_logger()

DEFAULT_ALPHA = 0.5
DEFAULT_THRESHOLD_PERCENTILE = 0.30


class ModelBFilter:
    """Race quality filter based on Model A + Model B combined score.

    Usage:
        filter = ModelBFilter.load(model_version_dir)
        if filter.should_bet(race_features, model_a_booster, model_a_features):
            # proceed with betting
    """

    def __init__(
        self,
        model_b: lgb.Booster,
        value_features: list[str],
        model_a_version: str,
        alpha: float = DEFAULT_ALPHA,
        threshold: float | None = None,
    ) -> None:
        self.model_b = model_b
        self.value_features = value_features
        self.model_a_version = model_a_version
        self.alpha = alpha
        self.threshold = threshold

    @classmethod
    def load(cls, version_dir: str | Path, *, alpha: float = DEFAULT_ALPHA) -> ModelBFilter | None:
        """Load Model B from a model version directory.

        Returns None if Model B files are not found.
        """
        version_dir = Path(version_dir)
        model_path = version_dir / "value_model_b.txt"
        meta_path = version_dir / "value_model_b_meta.pkl"
        threshold_path = version_dir / "model_b_threshold.json"

        if not model_path.exists() or not meta_path.exists():
            log.debug("model_b_not_found", dir=str(version_dir))
            return None

        import pickle

        model_b = lgb.Booster(model_file=str(model_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)  # noqa: S301

        threshold = None
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold_data = json.load(f)
            threshold = threshold_data.get("threshold")
            log.info("model_b_threshold_loaded", threshold=threshold)

        return cls(
            model_b=model_b,
            value_features=meta["value_features"],
            model_a_version=meta.get("model_a_version", ""),
            alpha=alpha,
            threshold=threshold,
        )

    def should_bet(
        self,
        race_features: pl.DataFrame,
        model_a_booster: lgb.Booster,
        model_a_features: list[str],
    ) -> tuple[bool, float]:
        """Determine if a race passes the Model B quality filter.

        Returns (should_bet, race_quality_score).
        If threshold is not calibrated, always returns True.
        """
        if self.threshold is None:
            return True, 0.0

        race_quality = self._compute_race_quality(race_features, model_a_booster, model_a_features)

        passes = race_quality >= self.threshold
        log.debug(
            "model_b_filter",
            race_quality=round(race_quality, 4),
            threshold=round(self.threshold, 4),
            passes=passes,
        )
        return passes, race_quality

    def compute_combined_scores(
        self,
        features: pl.DataFrame,
        model_a_booster: lgb.Booster,
        model_a_features: list[str],
    ) -> np.ndarray:
        """Compute per-runner combined scores for a race or batch of races.

        The features DataFrame may contain raw string categoricals (from DataLoader).
        We encode them before passing to Model A / Model B.
        """
        from providence.features.pipeline import FeaturePipeline

        encoded = FeaturePipeline._encode_categoricals(features)

        value_scores = predict_value_scores(
            self.model_b, encoded, model_a_booster, model_a_features, self.value_features,
        )

        from providence.model.value_model import _add_model_a_scores

        scored = _add_model_a_scores(encoded, model_a_booster, model_a_features)
        model_a_ranks = scored["model_a_rank"].cast(pl.Float64).to_numpy()
        n_runners = len(model_a_ranks)
        a_norm = 1.0 - (model_a_ranks - 1.0) / max(n_runners - 1, 1)

        b_min, b_max = value_scores.min(), value_scores.max()
        b_norm = (value_scores - b_min) / (b_max - b_min + 1e-12)

        return self.alpha * a_norm + (1.0 - self.alpha) * b_norm

    def _compute_race_quality(
        self,
        race_features: pl.DataFrame,
        model_a_booster: lgb.Booster,
        model_a_features: list[str],
    ) -> float:
        combined = self.compute_combined_scores(race_features, model_a_booster, model_a_features)
        return float(np.max(combined))

    @staticmethod
    def calibrate_threshold(
        features: pl.DataFrame,
        model_a_booster: lgb.Booster,
        model_a_features: list[str],
        model_b_filter: ModelBFilter,
        percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    ) -> float:
        """Compute the threshold from historical data.

        Args:
            features: Historical feature DataFrame (multiple races).
            percentile: The percentile below which races are filtered out.
                0.30 means keep top 70% of races.
        """
        race_ids = features["race_id"].unique(maintain_order=True).to_list()
        race_qualities = []

        for race_id in race_ids:
            race_df = features.filter(pl.col("race_id") == race_id)
            if race_df.height < 2:
                continue
            has_odds = "win_odds_rank" in race_df.columns and race_df["win_odds_rank"].is_not_null().any()
            if not has_odds:
                continue
            try:
                quality = model_b_filter._compute_race_quality(race_df, model_a_booster, model_a_features)
                race_qualities.append(quality)
            except Exception:  # noqa: BLE001
                continue

        if not race_qualities:
            log.warning("model_b_calibrate_no_data")
            return 0.0

        arr = np.array(race_qualities)
        threshold = float(np.percentile(arr, percentile * 100))
        log.info(
            "model_b_threshold_calibrated",
            n_races=len(race_qualities),
            percentile=percentile,
            threshold=round(threshold, 6),
            pass_rate=round(np.mean(arr >= threshold), 3),
        )
        return threshold

    def save_threshold(self, threshold: float, version_dir: str | Path) -> None:
        """Save the calibrated threshold to a JSON file."""
        path = Path(version_dir) / "model_b_threshold.json"
        with open(path, "w") as f:
            json.dump({"threshold": threshold, "alpha": self.alpha, "percentile": DEFAULT_THRESHOLD_PERCENTILE}, f)
        log.info("model_b_threshold_saved", path=str(path))
