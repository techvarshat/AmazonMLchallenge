from __future__ import annotations

from typing import Any, Dict, Iterable, List


class CrossEncoderReranker:
    """Small wrapper around a transformer-based reranker for ambiguous pairs.

    In production this could use a compact multilingual encoder such as
    facebook/xlm-roberta-base. The current implementation is intentionally
    lightweight and does not train a full model in this smoke test setup.
    """

    def __init__(self, model_name: str = "FacebookAI/xlm-roberta-base") -> None:
        self.model_name = model_name

    def fit(self, pairs: Iterable[Dict[str, Any]], labels: Iterable[int]) -> "CrossEncoderReranker":
        return self

    def predict_proba(self, pairs: Iterable[Dict[str, Any]]) -> List[float]:
        return [0.5 for _ in pairs]
