from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def evaluate_candidate_recall(true_matches: dict[str, set[str]], candidates: dict[str, set[str]]) -> dict[str, float | int]:
    total = 0
    hits = 0
    for source1_id, positives in true_matches.items():
        total += 1
        if positives and positives & candidates.get(source1_id, set()):
            hits += 1
        elif not positives and not candidates.get(source1_id):
            hits += 1
    coverage = hits / total if total else 0.0
    return {"candidate_recall": coverage, "total_entities": total, "covered_entities": hits}


def build_truth_index(ground_truth: Iterable[dict]) -> dict[str, set[str]]:
    truth = defaultdict(set)
    for row in ground_truth:
        s1 = str(row["source1_entity_id"]).strip()
        matched = row.get("matched_entity_ids", "")
        if not matched:
            truth[s1] = set()
            continue
        truth[s1] = {item.strip() for item in matched.split(",") if item.strip()}
    return dict(truth)
