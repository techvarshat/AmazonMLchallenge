from __future__ import annotations

import pandas as pd
import pytest

from src.lightgbm_pipeline import build_truth_map, build_training_rows, normalize_frame
from src.data_loader import load_ground_truth, load_source_table


def test_truth_map_preserves_valid_matches():
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "matched_entity_ids": ["S2-10,S3-11", ""],
        }
    )
    mapped = build_truth_map(truth)
    assert mapped["S1-1"] == {"S2-10", "S3-11"}
    assert mapped["S1-2"] == set()


def test_build_training_rows_keeps_positive_labels_when_matches_exist():
    source1 = pd.DataFrame(
        [
            {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Beta LLC", "business_address": "2 Elm St", "country": "US"},
        ]
    )
    target = pd.DataFrame(
        [
            {"entity_id": "S2-10", "business_name": "Acme Corporation", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "S2-99", "business_name": "Other Shop", "business_address": "99 Road", "country": "US"},
        ]
    )
    source1_norm = normalize_frame(source1)
    target_norm = normalize_frame(target)
    truth_map = {"S1-1": {"S2-10"}, "S1-2": set()}
    rows, labels = build_training_rows(source1_norm, target_norm, truth_map, max_candidates=10)
    assert len(rows) == len(labels)
    assert 1 in labels
    assert labels.count(1) > 0


def test_train_lgbm_raises_on_single_class():
    from src.train_lgbm import train_lightgbm_model

    with pytest.raises(ValueError, match="only one class"):
        train_lightgbm_model([{"a": 1}, {"a": 2}], [1, 1])


def test_limit_mismatch_errors_when_truth_and_target_do_not_overlap():
    source1 = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Alpha"], "business_address": ["1 St"], "country": ["US"]})
    target = pd.DataFrame({"entity_id": ["S2-2"], "business_name": ["Beta"], "business_address": ["2 St"], "country": ["US"]})
    truth = {"S1-1": {"S3-3"}}
    with pytest.raises(ValueError, match="--limit"):
        from src.lightgbm_pipeline import align_training_subset

        align_training_subset(normalize_frame(source1), normalize_frame(target), truth, limit=1)
