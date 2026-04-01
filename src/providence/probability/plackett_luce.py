"""Plackett-Luce probability utilities."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


def compute_win_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = _strengths(scores, temperature)
    return values / values.sum()


def compute_exacta_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[tuple[int, int], float]:
    values = _strengths(scores, temperature)
    total = values.sum()
    probs: dict[tuple[int, int], float] = {}
    for i in range(len(values)):
        p_i = values[i] / total
        remaining = total - values[i]
        for j in range(len(values)):
            if i == j:
                continue
            probs[(i, j)] = float(p_i * (values[j] / remaining))
    return probs


def compute_quinella_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[tuple[int, int], float]:
    exacta = compute_exacta_probs(scores, temperature)
    probs: dict[tuple[int, int], float] = {}
    n = len(scores)
    for i, j in combinations(range(n), 2):
        probs[(i, j)] = exacta[(i, j)] + exacta[(j, i)]
    return probs


def compute_trifecta_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[tuple[int, int, int], float]:
    values = _strengths(scores, temperature)
    total = values.sum()
    probs: dict[tuple[int, int, int], float] = {}
    n = len(values)
    for i, j, k in permutations(range(n), 3):
        p_i = values[i] / total
        rem1 = total - values[i]
        p_j = values[j] / rem1
        rem2 = rem1 - values[j]
        p_k = values[k] / rem2
        probs[(i, j, k)] = float(p_i * p_j * p_k)
    return probs


def compute_trio_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[tuple[int, int, int], float]:
    trifecta = compute_trifecta_probs(scores, temperature)
    probs: dict[tuple[int, int, int], float] = {}
    n = len(scores)
    for combo in combinations(range(n), 3):
        probs[combo] = float(sum(trifecta[p] for p in permutations(combo, 3)))
    return probs


def compute_wide_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[tuple[int, int], float]:
    trifecta = compute_trifecta_probs(scores, temperature)
    n = len(scores)
    probs: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(n), 2):
        total = 0.0
        for k in range(n):
            if k in (i, j):
                continue
            for perm in permutations((i, j, k), 3):
                total += trifecta[perm]
        probs[(i, j)] = float(total)
    return probs


def compute_all_ticket_probs(scores: np.ndarray, temperature: float = 1.0) -> dict[str, dict]:
    win = compute_win_probs(scores, temperature)
    exacta = compute_exacta_probs(scores, temperature)
    quinella = compute_quinella_probs(scores, temperature)
    trifecta = compute_trifecta_probs(scores, temperature)
    trio = compute_trio_probs(scores, temperature)
    wide = compute_wide_probs(scores, temperature)
    return {
        "win": {i: float(p) for i, p in enumerate(win)},
        "exacta": exacta,
        "quinella": quinella,
        "trifecta": trifecta,
        "trio": trio,
        "wide": wide,
    }


def _strengths(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(scores, dtype=float) / temperature
    scaled -= np.max(scaled)
    return np.exp(scaled)
