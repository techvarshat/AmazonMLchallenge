from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def train_lightgbm_model(feature_rows: Iterable[Dict[str, Any]], target: Iterable[int]) -> Any:
    if lgb is None:
        raise ImportError("lightgbm is not installed. Install dependencies from requirements.txt.")
    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.asarray(list(target), dtype=int)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)
    return model
