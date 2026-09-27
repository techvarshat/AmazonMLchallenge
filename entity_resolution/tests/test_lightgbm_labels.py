from __future__ import annotations

import pandas as pd
import pytest

from src.lightgbm_pipeline import (
    build_truth_map,
    build_training_rows,
    generate_candidate_map,
    load_limited_training_sample,
    normalize_frame,
    train_lightgbm_only_experiment,
)
from src.data_loader import load_ground_truth, load_source_table


def _write_small_train_dataset(path):
    train_dir = path / "dataset" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Zeta Labs", "business_address": "2 Oak Ave", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Gamma Tech", "business_address": "3 Pine Rd", "country": "CA"},
        ]
    ).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"entity_id": "T2-1", "business_name": "Acme Corporation", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "T2-2", "business_name": "Beta LLC", "business_address": "999 Side St", "country": "US"},
            {"entity_id": "T2-3", "business_name": "Zeta Laboratories", "business_address": "2 Oak Ave", "country": "US"},
        ]
    ).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"entity_id": "T3-4", "business_name": "Other Shop", "business_address": "4 Broadway", "country": "US"},
        ]
    ).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "T2-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "T2-3"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
        ]
    ).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)
    return train_dir


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


def test_limited_training_sample_keeps_truth_alignment(tmp_path):
    train_dir = _write_small_train_dataset(tmp_path)
    source1_df, target_df, truth_map = load_limited_training_sample(train_dir, limit=2)

    assert set(source1_df["entity_id"]) == {"S1-1", "S1-2"}
    assert set(truth_map) == {"S1-1", "S1-2"}
    assert {"T2-1", "T2-3"}.issubset(set(target_df["entity_id"]))


def test_generate_candidate_map_is_bounded_by_max_candidates():
    source1 = pd.DataFrame(
        [
            {"entity_id": "S1-1", "business_name": "Apple Inc", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Banana Co", "business_address": "2 Pine Rd", "country": "US"},
        ]
    )
    target = pd.DataFrame(
        [
            {"entity_id": "T1", "business_name": "Apple Inc", "business_address": "1 Main St", "country": "US"},
            {"entity_id": "T2", "business_name": "Apple LLC", "business_address": "2 Main St", "country": "US"},
            {"entity_id": "T3", "business_name": "Banana Company", "business_address": "2 Pine Rd", "country": "US"},
            {"entity_id": "T4", "business_name": "Banana Ltd", "business_address": "3 Pine Rd", "country": "US"},
        ]
    )
    source1_norm = normalize_frame(source1)
    target_norm = normalize_frame(target)
    candidate_map = generate_candidate_map(source1_norm, target_norm, max_candidates=2)
    assert all(len(values) <= 2 for values in candidate_map.values())


def test_train_lightgbm_only_experiment_smoke_runs_on_small_synthetic_data(tmp_path):
    train_dir = _write_small_train_dataset(tmp_path)
    result = train_lightgbm_only_experiment(
        data_root=tmp_path,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        max_candidates=10,
        limit=2,
    )
    assert result["n_training_rows"] > 0
    assert result["threshold"] > 0
    assert (tmp_path / "models" / "lightgbm_entity_resolution.joblib").exists()
