from __future__ import annotations

from typing import Any, Dict, Iterable, List


def select_best_matches(entity_candidates: Dict[str, set[str]], scores: Dict[str, Dict[str, float]], threshold: float = 0.5) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for source1_id, candidate_ids in entity_candidates.items():
        ranked = []
        for cand_id in sorted(candidate_ids):
            ranked.append((cand_id, scores.get(source1_id, {}).get(cand_id, 0.0)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        chosen = [cand_id for cand_id, score in ranked if score >= threshold]
        results[source1_id] = chosen[:1] if chosen else []
    return results
