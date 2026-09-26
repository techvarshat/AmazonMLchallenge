from __future__ import annotations

from typing import Dict, Iterable, List


def average_probabilities(probability_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, list[float]] = {}
    for record in probability_dicts:
        for key, value in record.items():
            merged.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in merged.items()}


def stack_probabilities(probability_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    return average_probabilities(probability_dicts)
