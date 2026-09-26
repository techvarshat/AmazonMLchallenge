from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def train_xgboost_model(feature_rows: Iterable[Dict[str, Any]], target: Iterable[int]) -> Any:
    if xgb is None:
        raise ImportError("xgboost is not installed. Install dependencies from requirements.txt.")
    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.asarray(list(target), dtype=int)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
    )
    model.fit(X, y)
    return model
