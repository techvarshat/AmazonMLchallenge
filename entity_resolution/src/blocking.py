from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def build_name_index(records: Iterable[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for row in records:
        key = str(row.get("name_norm", "")).strip().lower()
        if key:
            index[key].append(str(row["entity_id"]))
    return dict(index)


def build_address_index(records: Iterable[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for row in records:
        key = str(row.get("address_norm", "")).strip().lower()
        if key:
            index[key].append(str(row["entity_id"]))
    return dict(index)


def build_token_index(records: Iterable[dict]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in records:
        for token in row.get("tokens", []):
            if token:
                index[token].add(str(row["entity_id"]))
    return dict(index)


def generate_exact_candidates(source1_records: Iterable[dict], source2_records: Iterable[dict], source3_records: Iterable[dict]) -> dict[str, set[str]]:
    candidate_map: dict[str, set[str]] = defaultdict(set)
    target_records = list(source2_records) + list(source3_records)
    name_index = build_name_index(target_records)
    address_index = build_address_index(target_records)
    country_name_index = defaultdict(set)
    for row in target_records:
        key = f"{str(row.get('country','')).strip().lower()}::{str(row.get('name_norm','')).strip().lower()}"
        if key:
            country_name_index[key].add(str(row["entity_id"]))

    for left in source1_records:
        left_id = str(left["entity_id"])
        left_name = str(left.get("name_norm", "")).strip().lower()
        left_address = str(left.get("address_norm", "")).strip().lower()
        left_country = str(left.get("country", "")).strip().lower()
        country_name_key = f"{left_country}::{left_name}"

        candidate_map[left_id].update(name_index.get(left_name, []))
        candidate_map[left_id].update(address_index.get(left_address, []))
        candidate_map[left_id].update(country_name_index.get(country_name_key, []))

    return dict(candidate_map)


def generate_token_candidates(source1_records: Iterable[dict], source2_records: Iterable[dict], source3_records: Iterable[dict], max_candidates: int = 50) -> dict[str, set[str]]:
    target_records = list(source2_records) + list(source3_records)
    token_index = build_token_index(target_records)

    candidates: dict[str, set[str]] = defaultdict(set)
    for left in source1_records:
        left_id = str(left["entity_id"])
        for token in left.get("tokens", []):
            if not token:
                continue
            candidates[left_id].update(token_index.get(token, set()))
        if len(candidates[left_id]) > max_candidates:
            candidates[left_id] = set(list(candidates[left_id])[:max_candidates])
    return dict(candidates)


def merge_candidate_sets(*sets_by_entity: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for candidate_dict in sets_by_entity:
        for entity_id, cand_ids in candidate_dict.items():
            merged[entity_id].update(cand_ids)
    return dict(merged)
