from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_candidates(source1_records: Iterable[dict], target_records: Iterable[dict], top_k: int = 20, analyzer: str = "char", ngram_range=(2, 5)) -> dict[str, list[str]]:
    left_texts = [" ".join(r.get("name_tokens", []) + r.get("address_tokens", [])) for r in source1_records]
    right_texts = [" ".join(r.get("name_tokens", []) + r.get("address_tokens", [])) for r in target_records]

    if not left_texts or not right_texts:
        return {}

    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2)
    left_matrix = vectorizer.fit_transform(left_texts)
    right_matrix = vectorizer.transform(right_texts)
    sims = cosine_similarity(left_matrix, right_matrix)
    results: dict[str, list[str]] = {}
    for i, record in enumerate(source1_records):
        indices = sims[i].argsort()[::-1][:top_k]
        results[str(record["entity_id"])] = [str(target_records[j]["entity_id"]) for j in indices]
    return results


def compute_word_tfidf_candidates(source1_records: Iterable[dict], target_records: Iterable[dict], top_k: int = 15) -> dict[str, list[str]]:
    left_texts = [" ".join(r.get("name_tokens", [])[:30]) for r in source1_records]
    right_texts = [" ".join(r.get("name_tokens", [])[:30]) for r in target_records]
    if not left_texts or not right_texts:
        return {}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    left_matrix = vectorizer.fit_transform(left_texts)
    right_matrix = vectorizer.transform(right_texts)
    sims = cosine_similarity(left_matrix, right_matrix)
    results: dict[str, list[str]] = {}
    all_target = list(target_records)
    for i, record in enumerate(source1_records):
        indices = sims[i].argsort()[::-1][:top_k]
        results[str(record["entity_id"])] = [str(all_target[j]["entity_id"]) for j in indices]
    return results
