from __future__ import annotations

import numpy as np


def macro_f05_score(y_true, y_prob, threshold: float = 0.5):
    """Compute macro F0.5 score from binary labels and probability scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= threshold).astype(int)

    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) else 0.0
    return float(f05)
