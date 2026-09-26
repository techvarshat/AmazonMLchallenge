from __future__ import annotations

from typing import Dict, Iterable, List, Set


def mine_hard_negatives(candidate_pairs: Iterable[tuple[str, str, float]], positive_ids: Set[str], top_n: int = 5) -> List[tuple[str, str]]:
    """Select plausible negatives near the positive boundary.

    This is intentionally lightweight: the real pipeline would use fuzzy and
    embedding similarity to rank candidates and keep the most informative false
    pairs for retraining.
    """
    mined: List[tuple[str, str]] = []
    for left_id, right_id, score in candidate_pairs:
        if right_id not in positive_ids and score > 0.65:
            mined.append((left_id, right_id))
            if len(mined) >= top_n:
                break
    return mined
