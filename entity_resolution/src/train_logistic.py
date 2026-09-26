from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_model(feature_rows: Iterable[Dict[str, Any]], target: Iterable[int]) -> LogisticRegression:
    X = pd.DataFrame(feature_rows).fillna(0)
    y = list(target)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model
