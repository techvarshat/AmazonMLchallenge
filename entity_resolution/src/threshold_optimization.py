from __future__ import annotations

from typing import Iterable


def optimize_threshold(probabilities: Iterable[float], labels: Iterable[int], grid: Iterable[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_f05 = -1.0
    for threshold in grid:
        pred = [1 if p >= threshold else 0 for p in probabilities]
        tp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(pred, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(pred, labels) if p == 0 and y == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) else 0.0
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
    return best_threshold, best_f05
