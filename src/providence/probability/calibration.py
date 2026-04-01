"""Temperature scaling for probability calibration."""

from __future__ import annotations

import math

import numpy as np
import optuna

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
