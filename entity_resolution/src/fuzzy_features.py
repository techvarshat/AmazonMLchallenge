from __future__ import annotations

from typing import Any, Dict

from rapidfuzz import fuzz


def compute_fuzzy_features(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    left_name = str(left.get("business_name", ""))
    right_name = str(right.get("business_name", ""))
    left_addr = str(left.get("business_address", ""))
    right_addr = str(right.get("business_address", ""))

    features["rapidfuzz_ratio"] = fuzz.ratio(left_name, right_name)
    features["rapidfuzz_partial_ratio"] = fuzz.partial_ratio(left_name, right_name)
    features["rapidfuzz_token_sort_ratio"] = fuzz.token_sort_ratio(left_name, right_name)
    features["rapidfuzz_token_set_ratio"] = fuzz.token_set_ratio(left_name, right_name)
    features["rapidfuzz_qratio"] = fuzz.QRatio(left_name, right_name)
    features["rapidfuzz_wratio"] = fuzz.WRatio(left_name, right_name)
    features["address_ratio"] = fuzz.ratio(left_addr, right_addr)
    features["address_partial_ratio"] = fuzz.partial_ratio(left_addr, right_addr)
    return features
