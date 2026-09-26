from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def platt_scale(scores: Iterable[float], labels: Iterable[int]) -> LogisticRegression:
    X = np.asarray(list(scores), dtype=float).ravel().reshape(-1, 1)
    y = np.asarray(list(labels), dtype=int)
    model = LogisticRegression()
    model.fit(X, y)
    return model


def isotonic_scale(scores: Iterable[float], labels: Iterable[int]) -> IsotonicRegression:
    X = np.asarray(list(scores), dtype=float).ravel()
    y = np.asarray(list(labels), dtype=int)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(X, y)
    return model
