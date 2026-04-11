"""Probability calibration: temperature scaling and isotonic regression."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression

from providence.probability.plackett_luce import compute_win_probs


class TemperatureScaler:
    """Optimize a single temperature parameter for win probabilities."""

    def __init__(self, temperature: float = 1.0) -> None:
        self._temperature = temperature

    def fit(self, scores_per_race: list[np.ndarray], winners_per_race: list[int], n_trials: int = 50) -> float:
        if len(scores_per_race) != len(winners_per_race):
            raise ValueError("scores_per_race and winners_per_race must have the same length")

        def objective(trial: optuna.Trial) -> float:
            temperature = trial.suggest_float("temperature", 0.1, 10.0, log=True)
            losses = []
            for scores, winner_idx in zip(scores_per_race, winners_per_race, strict=True):
                probs = compute_win_probs(scores, temperature)
                p = max(float(probs[winner_idx]), 1e-12)
                losses.append(-math.log(p))
            return float(np.mean(losses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self._temperature = float(study.best_trial.params["temperature"])
        return self._temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float) / self._temperature


class IsotonicCalibrator:
    """Non-parametric probability calibration via isotonic regression.

    Learns the mapping from raw PL win probabilities to calibrated
    probabilities that better match observed win frequencies.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        self._fitted = False

    def fit(
        self,
        scores_per_race: list[np.ndarray],
        winners_per_race: list[int],
        temperature: float,
    ) -> None:
        raw_probs: list[float] = []
        labels: list[float] = []
        for scores, winner_idx in zip(scores_per_race, winners_per_race, strict=True):
            probs = compute_win_probs(scores, temperature)
            for i, p in enumerate(probs):
                raw_probs.append(float(p))
                labels.append(1.0 if i == winner_idx else 0.0)

        X = np.array(raw_probs, dtype=float)
        y = np.array(labels, dtype=float)
        self._model.fit(X, y)
        self._fitted = True

    def calibrate(self, win_probs: np.ndarray) -> np.ndarray:
        """Calibrate raw win probabilities and re-normalize to sum to 1."""
        if not self._fitted:
            return np.asarray(win_probs, dtype=float)
        calibrated = self._model.predict(np.asarray(win_probs, dtype=float))
        calibrated = np.maximum(calibrated, 1e-12)
        total = calibrated.sum()
        return calibrated / total if total > 0 else calibrated

    @property
    def fitted(self) -> bool:
        return self._fitted

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301
        self._fitted = True
