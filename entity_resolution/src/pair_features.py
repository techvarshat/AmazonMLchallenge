from __future__ import annotations

from math import isclose
from typing import Any, Dict

from rapidfuzz import fuzz


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def build_pair_features(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float | int | bool]:
    left_name = str(left.get("business_name", ""))
    right_name = str(right.get("business_name", ""))
    left_addr = str(left.get("business_address", ""))
    right_addr = str(right.get("business_address", ""))
    left_country = str(left.get("country", "")).strip()
    right_country = str(right.get("country", "")).strip()

    left_tokens = set(str(left.get("tokens", "")).split()) if isinstance(left.get("tokens", ""), str) else set(left.get("tokens", []) or [])
    right_tokens = set(str(right.get("tokens", "")).split()) if isinstance(right.get("tokens", ""), str) else set(right.get("tokens", []) or [])
    left_name_tokens = set(str(left.get("name_tokens", "")).split()) if isinstance(left.get("name_tokens", ""), str) else set(left.get("name_tokens", []) or [])
    right_name_tokens = set(str(right.get("name_tokens", "")).split()) if isinstance(right.get("name_tokens", ""), str) else set(right.get("name_tokens", []) or [])

    left_name_sorted = sorted(left_name_tokens)
    right_name_sorted = sorted(right_name_tokens)

    name_fuzzy = fuzz.ratio(left_name.lower(), right_name.lower())
    partial = fuzz.partial_ratio(left_name.lower(), right_name.lower())
    token_sort = fuzz.token_sort_ratio(left_name.lower(), right_name.lower())
    token_set = fuzz.token_set_ratio(left_name.lower(), right_name.lower())
    address_ratio = fuzz.ratio(left_addr.lower(), right_addr.lower())

    name_jaccard = _jaccard(left_name_tokens, right_name_tokens)
    token_jaccard = _jaccard(left_tokens, right_tokens)
    shared_tokens = len(left_tokens & right_tokens)

    features: Dict[str, float | int | bool] = {
        "name_exact_match": int(left_name == right_name),
        "normalized_name_match": int(str(left.get("name_norm", "")).lower() == str(right.get("name_norm", "")).lower()),
        "compact_name_match": int(str(left.get("name_compact", "")).lower() == str(right.get("name_compact", "")).lower()),
        "address_exact_match": int(left_addr == right_addr),
        "normalized_address_match": int(str(left.get("address_norm", "")).lower() == str(right.get("address_norm", "")).lower()),
        "country_exact_match": int(left_country == right_country),
        "country_missing_left": int(not left_country),
        "country_missing_right": int(not right_country),
        "both_country_missing": int(not left_country and not right_country),
        "country_conflict": int(bool(left_country and right_country and left_country != right_country)),
        "rapidfuzz_ratio": float(name_fuzzy),
        "rapidfuzz_partial_ratio": float(partial),
        "rapidfuzz_token_sort_ratio": float(token_sort),
        "rapidfuzz_token_set_ratio": float(token_set),
        "address_ratio": float(address_ratio),
        "name_len_diff": abs(len(left_name) - len(right_name)),
        "address_len_diff": abs(len(left_addr) - len(right_addr)),
        "name_jaccard": float(name_jaccard),
        "token_jaccard": float(token_jaccard),
        "shared_tokens": float(shared_tokens),
        "first_token_match": int(bool(left_name_sorted and right_name_sorted and left_name_sorted[0] == right_name_sorted[0])),
        "last_token_match": int(bool(left_name_sorted and right_name_sorted and left_name_sorted[-1] == right_name_sorted[-1])),
    }
    return features
