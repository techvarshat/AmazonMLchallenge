from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set


def build_exact_index(records: Iterable[dict]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        entity_id = str(record["entity_id"])
        key = str(record.get("name_normalized", "")).strip()
        if key:
            index[key].append(entity_id)
    return dict(index)


def build_address_index(records: Iterable[dict]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        entity_id = str(record["entity_id"])
        key = str(record.get("address_normalized", "")).strip()
        if key:
            index[key].append(entity_id)
    return dict(index)


def build_token_index(records: Iterable[dict]) -> Dict[str, Set[str]]:
    token_index: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        tokens = record.get("tokens", [])
        entity_id = str(record["entity_id"])
        for token in tokens:
            token_index[token].add(entity_id)
    return dict(token_index)


def candidate_union(*candidate_sets: Iterable[set]) -> set:
    union: set = set()
    for candidate_set in candidate_sets:
        union |= set(candidate_set)
    return union
