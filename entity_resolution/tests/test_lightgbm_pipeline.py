from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.lightgbm_pipeline import build_truth_map, load_config, resolve_data_paths
from src.submission import write_candidate_pairs, write_matching_results


def test_load_config_defaults():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    assert cfg["project"]["name"] == "entity_resolution"
    assert cfg["models"]["use_lightgbm"] is True


def test_resolve_data_paths_uses_drive_root_when_available(tmp_path):
    drive_root = tmp_path / "drive" / "amazon_ml"
    raw_dir = drive_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    cfg = {
        "paths": {
            "drive_root": str(drive_root),
            "raw_data_dir": str(raw_dir),
            "train_dir": str(raw_dir),
            "test_dir": str(raw_dir),
        }
    }
    resolved = resolve_data_paths(cfg, data_root=None)
    assert resolved["train_dir"] == raw_dir
    assert resolved["test_dir"] == raw_dir


def test_truth_map_builds_empty_set_for_missing_matches():
    truth = pd.DataFrame(
        [{"source1_entity_id": "S1-1", "matched_entity_ids": ""}, {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-9,S3-10"}]
    )
    mapped = build_truth_map(truth)
    assert mapped["S1-1"] == set()
    assert mapped["S1-2"] == {"S2-9", "S3-10"}


def test_submission_schema_and_headers(tmp_path):
    out_dir = tmp_path / "outputs"
    candidates = {"S1-1": ["S2-2", "S3-3"], "S1-2": ["S2-2"]}
    matches = {"S1-1": ["S2-2"], "S1-2": []}
    write_candidate_pairs(out_dir / "candidate_pairs.tsv", candidates)
    write_matching_results(out_dir / "matching_results.tsv", matches)

    cand = (out_dir / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()[0]
    match = (out_dir / "matching_results.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert cand == "source1_entity_id\tcandidate_entity_ids"
    assert match == "source1_entity_id\tmatched_entity_ids"
