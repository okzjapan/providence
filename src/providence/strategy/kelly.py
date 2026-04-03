"""Scenario-based Kelly optimization helpers."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from providence.domain.enums import TicketType
from providence.strategy.types import RacePredictionBundle, TicketCandidate


def enumerate_top3_scenarios(bundle: RacePredictionBundle) -> list[tuple[tuple[int, int, int], float]]:
    values = _strengths(bundle.scores, bundle.temperature)
    total = values.sum()
    scenarios: list[tuple[tuple[int, int, int], float]] = []
    for i, j, k in permutations(range(len(values)), 3):
        p_i = values[i] / total
        rem1 = total - values[i]
        p_j = values[j] / rem1
        rem2 = rem1 - values[j]
        p_k = values[k] / rem2
        post_positions = (
            bundle.index_map.post_position_for_index(i),
            bundle.index_map.post_position_for_index(j),
            bundle.index_map.post_position_for_index(k),
        )
        scenarios.append((post_positions, float(p_i * p_j * p_k)))
    return scenarios


def scenario_return_matrix(
    candidates: list[TicketCandidate],
    scenarios: list[tuple[tuple[int, int, int], float]],
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray([prob for _, prob in scenarios], dtype=float)
    matrix = np.zeros((len(scenarios), len(candidates)), dtype=float)

    for scenario_idx, (top3, _) in enumerate(scenarios):
        for candidate_idx, candidate in enumerate(candidates):
            hit = _ticket_hits(candidate.ticket_type, candidate.combination, top3)
            matrix[scenario_idx, candidate_idx] = (candidate.odds_value - 1.0) if hit else -1.0
    return probs, matrix


def optimize_kelly_fractions(
    *,
    candidates: list[TicketCandidate],
    bundle: RacePredictionBundle,
    race_cap_fraction: float,
    max_iter: int = 200,
) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=float)

    scenarios = enumerate_top3_scenarios(bundle)
    probs, returns = scenario_return_matrix(candidates, scenarios)
    weights = _initial_weights(candidates, race_cap_fraction)

    for step_idx in range(max_iter):
        wealth = 1.0 + returns @ weights
        if np.any(wealth <= 1e-9):
            weights *= 0.5
            continue

        gradient = (probs[:, None] * returns / wealth[:, None]).sum(axis=0)
        step = 0.1 / (1.0 + step_idx / 20.0)
        proposal = weights + step * gradient
        proposal = np.maximum(proposal, 0.0)
        total = proposal.sum()
        if total > race_cap_fraction and total > 0:
            proposal *= race_cap_fraction / total

        if np.allclose(proposal, weights, atol=1e-8):
            break
        weights = proposal

    return weights


def _initial_weights(candidates: list[TicketCandidate], race_cap_fraction: float) -> np.ndarray:
    weights = np.asarray([_single_ticket_kelly(c.probability, c.odds_value) for c in candidates], dtype=float)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.zeros(len(candidates), dtype=float)
    if total > race_cap_fraction:
        weights *= race_cap_fraction / total
    return weights


def _single_ticket_kelly(probability: float, odds_value: float) -> float:
    b = odds_value - 1.0
    if b <= 0:
        return 0.0
    return max((probability * odds_value - 1.0) / b, 0.0)


def _ticket_hits(ticket_type: TicketType, combination: tuple[int, ...], top3: tuple[int, int, int]) -> bool:
    first, second, third = top3
    if ticket_type == TicketType.WIN:
        return combination == (first,)
    if ticket_type == TicketType.EXACTA:
        return combination == (first, second)
    if ticket_type == TicketType.QUINELLA:
        return tuple(sorted(combination)) == tuple(sorted((first, second)))
    if ticket_type == TicketType.WIDE:
        return set(combination).issubset(set(top3))
    if ticket_type == TicketType.TRIFECTA:
        return combination == top3
    if ticket_type == TicketType.TRIO:
        return tuple(sorted(combination)) == tuple(sorted(top3))
    raise ValueError(f"Unsupported ticket type: {ticket_type}")


def _strengths(scores: tuple[float, ...], temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=float) / temperature
    scaled -= scaled.max()
    return np.exp(scaled)
