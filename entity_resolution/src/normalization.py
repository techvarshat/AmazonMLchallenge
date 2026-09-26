from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from unidecode import unidecode


BUSINESS_SUFFIXES = {
    "pvt", "private", "limited", "ltd", "llp", "inc", "corp", "corporation",
    "llc", "co", "company", "cooperative", "sa", "sas", "sarl", "sci", "group",
    "gmbh", "plc", "bv", "ag", "pte", "pvt ltd", "private limited",
}


def normalize_text(value: str | None, lower: bool = True) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    if lower:
        text = text.lower().strip()
    return text


def compact_text(value: str | None) -> str:
    text = normalize_text(value)
    return re.sub(r"\s+", "", text)


def strip_business_suffixes(value: str | None) -> str:
    text = normalize_text(value)
    tokens = text.split()
    filtered = []
    for token in tokens:
        if token in BUSINESS_SUFFIXES:
            continue
        filtered.append(token)
    return " ".join(filtered)


def token_list(value: str | None) -> List[str]:
    text = normalize_text(value)
    return [tok for tok in text.split() if tok]


def build_feature_views(record: Dict[str, str]) -> Dict[str, str | List[str]]:
    name = record.get("business_name", "")
    address = record.get("business_address", "")
    country = record.get("country", "")
    views = {}
    views["name_raw"] = name
    views["name_lower"] = normalize_text(name, lower=True)
    views["name_ascii"] = unidecode(str(name)).lower().strip()
    views["name_compact"] = compact_text(name)
    views["name_tokens"] = token_list(name)
    views["name_no_suffix"] = strip_business_suffixes(name)
    views["address_raw"] = address
    views["address_lower"] = normalize_text(address, lower=True)
    views["address_ascii"] = unidecode(str(address)).lower().strip()
    views["address_compact"] = compact_text(address)
    views["address_tokens"] = token_list(address)
    views["country_raw"] = country
    views["country_normalized"] = normalize_text(country, lower=True)
    return views


def normalize_record(record: Dict[str, str]) -> Dict[str, str | List[str]]:
    return build_feature_views(record)
