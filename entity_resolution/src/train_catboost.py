from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


def train_catboost_model(feature_rows: Iterable[Dict[str, Any]], target: Iterable[int]) -> Any:
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed. Install dependencies from requirements.txt.")
    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.asarray(list(target), dtype=int)
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )
    model.fit(X, y)
    return model
