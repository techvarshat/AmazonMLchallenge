from __future__ import annotations

from typing import Any, Dict


def compute_embedding_features(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    """Placeholder for multilingual embedding similarity features.

    The actual production pipeline will use sentence-transformers or E5/BGE
    embeddings and FAISS lookup, but this function stays lightweight for
    project structure and smoke tests.
    """
    return {
        "e5_name_similarity": 0.0,
        "e5_address_similarity": 0.0,
        "e5_combined_similarity": 0.0,
        "bge_dense_similarity": 0.0,
    }
