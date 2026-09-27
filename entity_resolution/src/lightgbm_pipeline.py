from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from .blocking import generate_exact_candidates, generate_token_candidates, merge_candidate_sets
from .data_loader import load_ground_truth, load_source_table
from .normalization import compact_text, normalize_text, strip_business_suffixes, token_list
from .pair_features import build_pair_features
from .submission import write_candidate_pairs, write_matching_results
from .threshold_optimization import optimize_threshold
from .train_lgbm import train_lightgbm_model
from .validation import validate_candidate_file, validate_matching_file


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with Path(config_path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return config


def build_truth_map(truth_df: pd.DataFrame) -> dict[str, set[str]]:
    truth_map: dict[str, set[str]] = {}
    for row in truth_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        matched = row.matched_entity_ids
        if pd.isna(matched) or str(matched).strip() == "":
            truth_map[s1_id] = set()
            continue
        truth_map[s1_id] = {token.strip() for token in str(matched).split(",") if token.strip()}
    return truth_map


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].fillna("").map(normalize_text)
    out["address_norm"] = out["business_address"].fillna("").map(normalize_text)
    out["name_compact"] = out["business_name"].fillna("").map(compact_text)
    out["name_tokens"] = out["name_norm"].map(token_list)
    out["tokens"] = out.apply(lambda row: token_list(row["business_name"]) + token_list(row["business_address"]), axis=1)
    out["country_norm"] = out["country"].fillna("").map(lambda value: normalize_text(str(value)))
    out["name_no_suffix"] = out["business_name"].fillna("").map(strip_business_suffixes)
    return out


def _iter_dict_records(frame: pd.DataFrame) -> Any:
    for row in frame.itertuples(index=False):
        values = row._asdict()
        yield {key: value for key, value in values.items()}


def estimate_training_memory_mb(*frames: pd.DataFrame) -> float:
    total_bytes = 0
    for frame in frames:
        total_bytes += int(frame.memory_usage(index=True, deep=True).sum())
    return total_bytes / (1024 * 1024)


def load_limited_target_subset(
    train_dir: str | Path,
    truth_map: dict[str, set[str]],
    max_negative_ids: int = 250,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    positive_target_ids = {str(entity_id) for entity_ids in truth_map.values() for entity_id in entity_ids}
    negative_ids: list[str] = []
    seen_negative = set()

    for file_name in (Path(train_dir) / "train_source2.tsv", Path(train_dir) / "train_source3.tsv"):
        for chunk in pd.read_csv(
            file_name,
            sep="\t",
            engine="python",
            on_bad_lines="skip",
            dtype={"entity_id": "string"},
            chunksize=max(1, int(chunk_size)),
        ):
            for entity_id in chunk["entity_id"].astype(str).tolist():
                entity_id = str(entity_id)
                if entity_id in positive_target_ids or entity_id in seen_negative:
                    continue
                negative_ids.append(entity_id)
                seen_negative.add(entity_id)
                if len(negative_ids) >= max_negative_ids:
                    break
            if len(negative_ids) >= max_negative_ids:
                break
        if len(negative_ids) >= max_negative_ids:
            break

    selected_ids = set(positive_target_ids) | set(negative_ids)
    if not selected_ids:
        raise ValueError("No target IDs are available for limited training; the ground-truth target overlap is empty.")

    frames: list[pd.DataFrame] = []
    for file_name in (Path(train_dir) / "train_source2.tsv", Path(train_dir) / "train_source3.tsv"):
        for chunk in pd.read_csv(
            file_name,
            sep="\t",
            engine="python",
            on_bad_lines="skip",
            dtype={"entity_id": "string"},
            chunksize=max(1, int(chunk_size)),
        ):
            filtered = chunk[chunk["entity_id"].astype(str).isin(selected_ids)].copy()
            if not filtered.empty:
                frames.append(filtered)

    if not frames:
        raise ValueError("No target rows remained after bounded sampling for the limited training pass.")
    return pd.concat(frames, ignore_index=True)


def load_limited_training_sample(train_dir: str | Path, limit: int, max_negative_ids: int = 250) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    truth_df = load_ground_truth(Path(train_dir) / "train_ground_truth.tsv")
    ordered_ids = list(dict.fromkeys(truth_df["source1_entity_id"].astype(str).tolist()))[:limit]
    if not ordered_ids:
        raise ValueError("No source1 IDs were found in the ground-truth set for the bounded train sample.")

    truth_df = truth_df[truth_df["source1_entity_id"].astype(str).isin(ordered_ids)].copy()
    truth_map = build_truth_map(truth_df)
    source1_df = load_source_table(Path(train_dir) / "train_source1.tsv", selected_entity_ids=ordered_ids, chunk_size=100_000)
    if source1_df.empty:
        raise ValueError("The bounded source1 sample is empty; the selected source1 IDs are missing from the TSV.")
    source1_df = normalize_frame(source1_df)
    target_df = load_limited_target_subset(train_dir, truth_map, max_negative_ids=max_negative_ids)
    return source1_df, normalize_frame(target_df), truth_map


def generate_candidate_map(source1_df: pd.DataFrame, target_df: pd.DataFrame, max_candidates: int = 30) -> dict[str, set[str]]:
    exact = generate_exact_candidates(_iter_dict_records(source1_df), _iter_dict_records(target_df), ())
    token = generate_token_candidates(_iter_dict_records(source1_df), _iter_dict_records(target_df), (), max_candidates=max_candidates)
    merged = merge_candidate_sets(exact, token)
    output: dict[str, set[str]] = {}
    for entity_id, candidate_ids in merged.items():
        limited = set(list(candidate_ids)[:max_candidates])
        output[str(entity_id)] = limited
    return output


def build_training_rows(source1_df: pd.DataFrame, target_df: pd.DataFrame, truth_map: dict[str, set[str]], max_candidates: int = 30) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    candidate_map = generate_candidate_map(source1_df, target_df, max_candidates=max_candidates)
    target_by_id: dict[str, dict[str, Any]] = {}
    for row in target_df.itertuples(index=False):
        values = row._asdict()
        target_by_id[str(values["entity_id"])] = {key: value for key, value in values.items()}

    feature_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    groups: list[str] = []

    for source1_id, row in source1_df.set_index("entity_id").iterrows():
        left = row.to_dict()
        left["tokens"] = row.get("tokens", [])
        left["name_norm"] = row.get("name_norm", "")
        left["name_compact"] = row.get("name_compact", "")
        left["address_norm"] = row.get("address_norm", "")
        left["name_tokens"] = row.get("name_tokens", [])
        candidate_ids = candidate_map.get(str(source1_id), set())
        for candidate_id in sorted(candidate_ids):
            right = target_by_id.get(str(candidate_id))
            if right is None:
                continue
            right = dict(right)
            right["tokens"] = right.get("tokens", [])
            right["name_norm"] = right.get("name_norm", "")
            right["name_compact"] = right.get("name_compact", "")
            right["address_norm"] = right.get("address_norm", "")
            right["name_tokens"] = right.get("name_tokens", [])
            feature_rows.append(build_pair_features(left, right))
            labels.append(int(str(candidate_id) in truth_map.get(str(source1_id), set())))
            groups.append(str(source1_id))

    if not feature_rows:
        raise ValueError("No positive or negative training rows were generated from the challenge data.")
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "Training labels contain only one class after candidate generation. "
            "This usually means the selected subset or candidate-generation window excluded valid matches; "
            "check the truth-map to candidate-id overlap before training."
        )
    return feature_rows, labels, groups


def align_training_subset(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    truth_map: dict[str, set[str]],
    limit: int | None = None,
    max_negative_ids: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    if limit is None:
        return source1_df, target_df, truth_map

    truth_keys = [s1_id for s1_id in truth_map if str(s1_id) in source1_df["entity_id"].astype(str).tolist()]
    selected_s1_ids = truth_keys[:limit]
    if not selected_s1_ids:
        raise ValueError("No ground-truth source1 rows are available in the selected subset.")

    selected_truth_map = {str(s1_id): set(truth_map[str(s1_id)]) for s1_id in selected_s1_ids}
    valid_target_ids = set().union(*selected_truth_map.values()) if selected_truth_map else set()
    valid_target_ids = {str(target_id) for target_id in valid_target_ids if str(target_id) in set(target_df["entity_id"].astype(str))}
    if not valid_target_ids:
        raise ValueError(
            "The sampled truth rows do not overlap with the sampled target tables. "
            "This is why a --limit run can create all-negative labels. "
            "Keep the source1 truth subset aligned with the matching target IDs or remove the limit."
        )

    negative_pool = [str(candidate_id) for candidate_id in target_df["entity_id"].astype(str).tolist() if candidate_id not in valid_target_ids]
    negative_ids = set(negative_pool[:max_negative_ids])
    selected_target_ids = valid_target_ids | negative_ids

    selected_source1 = source1_df[source1_df["entity_id"].astype(str).isin(selected_s1_ids)].copy()
    selected_target = target_df[target_df["entity_id"].astype(str).isin(selected_target_ids)].copy()
    return selected_source1, selected_target, selected_truth_map


def resolve_data_paths(config: dict[str, Any], data_root: str | Path | None = None) -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    if data_root is not None:
        data_root = Path(data_root)
        if data_root.is_file():
            data_root = data_root.parent
        if "dataset" in data_root.parts:
            train_dir = data_root / "train"
            test_dir = data_root / "test"
        elif data_root.name == "raw":
            train_dir = data_root
            test_dir = data_root
        else:
            train_dir = data_root / "dataset" / "train"
            test_dir = data_root / "dataset" / "test"
        return {"project_root": project_root, "data_root": data_root, "train_dir": train_dir, "test_dir": test_dir}

    paths = config.get("paths", {})
    drive_root = Path(paths.get("drive_root", "/content/drive/MyDrive/amazon_ml"))
    raw_data_dir = Path(paths.get("raw_data_dir", drive_root / "data" / "raw"))
    if raw_data_dir.exists():
        return {
            "project_root": project_root,
            "data_root": raw_data_dir,
            "train_dir": raw_data_dir,
            "test_dir": raw_data_dir,
        }

    dataset_root = Path(paths.get("dataset_root", project_root / "data"))
    train_dir = Path(paths.get("train_dir", dataset_root / "dataset" / "train"))
    test_dir = Path(paths.get("test_dir", dataset_root / "dataset" / "test"))
    return {"project_root": project_root, "data_root": dataset_root, "train_dir": train_dir, "test_dir": test_dir}


def save_lightgbm_artifacts(
    model: Any,
    feature_columns: list[str],
    threshold: float,
    validation_f05: float,
    output_dir: str | Path,
    artifact_dir: str | Path | None = None,
    model_name: str = "lightgbm_entity_resolution.joblib",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir
    if artifact_dir is None:
        artifact_dir = output_dir.parent / "artifacts"
    artifact_dir = Path(artifact_dir)
    (artifact_dir / "metadata").mkdir(parents=True, exist_ok=True)

    model_path = model_dir / model_name
    feature_path = model_dir / "lightgbm_feature_columns.json"
    threshold_path = model_dir / "lightgbm_threshold.json"
    metadata_path = model_dir / "lightgbm_metadata.json"
    artifact_metadata_path = artifact_dir / "metadata" / "threshold_metadata.json"
    feature_schema_path = artifact_dir / "metadata" / "feature_schema.json"

    joblib.dump(model, model_path)
    feature_schema = {"feature_columns": feature_columns, "n_features": len(feature_columns)}
    feature_schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    feature_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    threshold_payload = {"threshold": float(threshold), "f05_validation": float(validation_f05)}
    threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    artifact_metadata_path.write_text(json.dumps({
        "threshold_status": "validated",
        "threshold_value": float(threshold),
        "validation_f05": float(validation_f05),
        "model_name": model_name,
        "notes": "Selected from the validation split only.",
    }, indent=2), encoding="utf-8")

    metadata = {
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "threshold_path": str(threshold_path),
        "metadata_path": str(metadata_path),
        "artifact_metadata_path": str(artifact_metadata_path),
        "feature_schema_path": str(feature_schema_path),
        "threshold": float(threshold),
        "validation_f05": float(validation_f05),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_lightgbm_only_experiment(
    data_root: str | Path | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
    model_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    max_candidates: int = 30,
    limit: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    resolved = resolve_data_paths(config, data_root=data_root)
    train_dir = resolved["train_dir"]
    test_dir = resolved["test_dir"]
    project_root = resolved["project_root"]

    if model_dir is None:
        model_dir = project_root / "models"
    if artifact_dir is None:
        artifact_dir = project_root / "artifacts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not (train_dir / "train_ground_truth.tsv").exists():
        raise FileNotFoundError(f"Missing ground-truth file in {train_dir}")

    print("[1/6] Loading training data")
    if limit is not None:
        print(f"Using bounded training sample with limit={limit} and deterministic source1/ground-truth alignment.")
        source1_df, target_df, truth_map = load_limited_training_sample(train_dir, limit=limit)
        print(f"Loaded bounded S1 rows={len(source1_df)}, target rows={len(target_df)}, truth entries={len(truth_map)}")
        print("Target pool is bounded to positive matches plus a deterministic negative sample to reduce Colab memory pressure.")
    else:
        truth_df = load_ground_truth(train_dir / "train_ground_truth.tsv")
        truth_map = build_truth_map(truth_df)
        source1_df = normalize_frame(load_source_table(train_dir / "train_source1.tsv"))
        source2_df = normalize_frame(load_source_table(train_dir / "train_source2.tsv"))
        source3_df = normalize_frame(load_source_table(train_dir / "train_source3.tsv"))
        target_df = pd.concat([source2_df, source3_df], ignore_index=True)
        mem_mb = estimate_training_memory_mb(source1_df, target_df, truth_df)
        print(f"Loaded full training data: S1={len(source1_df)}, target={len(target_df)}, truth entries={len(truth_map)}")
        print(f"Estimated memory footprint for training inputs: {mem_mb:.1f} MB")
        if mem_mb > 1024:
            print("Warning: the full-data training path may exceed Colab RAM. Use --limit for bounded smoke tests or a chunked target selection strategy in a higher-memory environment.")

    print("[2/6] Building candidate pairs")
    rows, labels, groups = build_training_rows(source1_df, target_df, truth_map, max_candidates=max_candidates)
    print(f"Generated {len(rows)} candidate pairs; positive={sum(labels)}, negative={len(labels)-sum(labels)}")
    if sum(labels) == 0 or len(set(labels)) < 2:
        raise ValueError("Single-class label set after candidate generation. This is a data-alignment issue, not a LightGBM issue.")

    print("[3/6] Preparing feature matrix and validation split")
    feature_frame = pd.DataFrame(rows).fillna(0)
    feature_columns = list(feature_frame.columns)
    y = np.asarray(labels, dtype=int)
    groups_array = np.asarray(groups)

    if len(groups_array) != len(y):
        raise ValueError("Group count does not match the number of training rows.")

    if len(np.unique(groups_array)) < 2:
        raise ValueError("Not enough distinct source1 entities for grouped validation.")

    splitter = GroupShuffleSplit(
        n_splits=30,
        test_size=0.2,
        random_state=42,
    )

    split_found = False
    for train_idx, val_idx in splitter.split(
        feature_frame,
        y,
        groups=groups_array,
    ):
        if (
            len(np.unique(y[train_idx])) == 2
            and len(np.unique(y[val_idx])) == 2
        ):
            split_found = True
            break

    if not split_found:
        raise ValueError(
            "Could not create a grouped validation split containing "
            "both classes in training and validation. "
            "Increase the sample size."
        )

    assert set(groups_array[train_idx]).isdisjoint(set(groups_array[val_idx]))

    X_train = feature_frame.iloc[train_idx].copy()
    X_val = feature_frame.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(
        f"Grouped split: train groups={len(np.unique(groups_array[train_idx]))}, "
        f"validation groups={len(np.unique(groups_array[val_idx]))}"
    )
    print(f"Train positives={int(y_train.sum())}, validation positives={int(y_val.sum())}")

    print("[4/6] Training LightGBM model")
    model = train_lightgbm_model(X_train.to_dict("records"), y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold_grid = config.get("training", {}).get("threshold_grid", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    if np.unique(y_val).size < 2:
        print("Warning: validation labels contain only one class; falling back to a default threshold of 0.5.")
        best_threshold = 0.5
        best_f05 = 0.0
    else:
        best_threshold, best_f05 = optimize_threshold(val_prob, y_val, threshold_grid)
    print(f"Validation F0.5={best_f05:.4f} with threshold={best_threshold:.3f}")

    print("[5/6] Saving model artifacts")
    save_lightgbm_artifacts(model, feature_columns, best_threshold, best_f05, model_dir, artifact_dir)

    print("[6/6] Finalizing training summary")
    return {
        "project_root": str(project_root),
        "data_root": str(data_root) if data_root is not None else str(resolved["data_root"]),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "validation_f05": float(best_f05),
        "threshold": float(best_threshold),
        "feature_columns": feature_columns,
        "model_path": str(model_dir / "lightgbm_entity_resolution.joblib"),
        "threshold_path": str(model_dir / "lightgbm_threshold.json"),
        "artifact_metadata_path": str(Path(artifact_dir) / "metadata" / "threshold_metadata.json"),
        "n_training_rows": int(len(feature_frame)),
    }


def run_lightgbm_inference(
    data_root: str | Path | None = None,
    output_dir: str | Path = "outputs",
    model_path: str | Path | None = None,
    threshold: float | None = None,
    model_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    validate_outputs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    resolved = resolve_data_paths(config, data_root=data_root)
    if data_root is None:
        data_root = resolved["data_root"]
    if model_dir is None:
        model_dir = resolved["project_root"] / "models"
    if artifact_dir is None:
        artifact_dir = resolved["project_root"] / "artifacts"
    model_dir = Path(model_dir)
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        model_path = model_dir / "lightgbm_entity_resolution.joblib"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained LightGBM model not found at {model_path}")

    if threshold is None:
        threshold_json = model_dir / "lightgbm_threshold.json"
        if threshold_json.exists():
            threshold = float(json.loads(threshold_json.read_text(encoding="utf-8"))["threshold"])
        else:
            threshold_file = artifact_dir / "metadata" / "threshold_metadata.json"
            if threshold_file.exists():
                threshold = float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold_value"])
            else:
                threshold = 0.5

    feature_columns_path = model_dir / "lightgbm_feature_columns.json"
    if feature_columns_path.exists():
        feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))["feature_columns"]
    else:
        feature_columns = None

    test_dir = resolved["test_dir"]
    source1_df = normalize_frame(load_source_table(test_dir / "test_source1.tsv"))
    source2_df = normalize_frame(load_source_table(test_dir / "test_source2.tsv"))
    source3_df = normalize_frame(load_source_table(test_dir / "test_source3.tsv"))
    target_df = pd.concat([source2_df, source3_df], ignore_index=True)
    candidate_map = generate_candidate_map(source1_df, target_df, max_candidates=30)

    model = joblib.load(model_path)
    scores_by_entity: dict[str, dict[str, float]] = {}
    candidate_pairs: dict[str, list[str]] = {}
    selected_matches: dict[str, list[str]] = {}

    for source1_id, candidate_ids in candidate_map.items():
        candidate_pairs[str(source1_id)] = sorted(candidate_ids)
        scores = {}
        for candidate_id in candidate_ids:
            match_row = target_df[target_df["entity_id"].astype(str) == str(candidate_id)]
            if match_row.empty:
                continue
            left = source1_df[source1_df["entity_id"].astype(str) == str(source1_id)].iloc[0].to_dict()
            right = match_row.iloc[0].to_dict()
            left["tokens"] = left.get("tokens", [])
            left["name_norm"] = left.get("name_norm", "")
            left["name_compact"] = left.get("name_compact", "")
            left["address_norm"] = left.get("address_norm", "")
            left["name_tokens"] = left.get("name_tokens", [])
            right["tokens"] = right.get("tokens", [])
            right["name_norm"] = right.get("name_norm", "")
            right["name_compact"] = right.get("name_compact", "")
            right["address_norm"] = right.get("address_norm", "")
            right["name_tokens"] = right.get("name_tokens", [])
            feature_row = build_pair_features(left, right)
            if feature_columns is not None:
                ordered = {key: feature_row.get(key, 0) for key in feature_columns}
            else:
                ordered = feature_row
            score = float(model.predict_proba(pd.DataFrame([ordered]).fillna(0))[0, 1])
            scores[str(candidate_id)] = score
        scores_by_entity[str(source1_id)] = scores

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected = [candidate for candidate, score in ranked if score >= float(threshold)]
        selected_matches[str(source1_id)] = selected

    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    write_candidate_pairs(candidate_path, candidate_pairs)
    write_matching_results(matching_path, selected_matches)

    if validate_outputs:
        test_ids = [str(value) for value in source1_df["entity_id"].astype(str).tolist()]
        validate_candidate_file(candidate_path, test_ids)
        validate_matching_file(matching_path, test_ids)

    summary = {
        "data_root": str(data_root),
        "model_path": str(model_path),
        "threshold": float(threshold),
        "candidate_pairs_path": str(candidate_path),
        "matching_results_path": str(matching_path),
        "n_source1_rows": int(len(source1_df)),
        "n_candidates": int(sum(len(values) for values in candidate_pairs.values())),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run the LightGBM-only entity resolution pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--data-root", type=str, default=None, help="Raw TSV directory or local student_resource root.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for generated submission files.")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory for saved model assets.")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Directory for metadata artifacts.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Run training or inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for supervised smoke tests.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional probability threshold used for inference.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a saved LightGBM model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        result = train_lightgbm_only_experiment(
            data_root=args.data_root,
            config_path=args.config,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            artifact_dir=args.artifact_dir,
            limit=args.limit,
        )
        print(json.dumps({"status": "training_complete", **result}, indent=2))
        return

    result = run_lightgbm_inference(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        model_dir=args.model_dir,
        artifact_dir=args.artifact_dir,
        config_path=args.config,
    )
    print(json.dumps({"status": "inference_complete", **result}, indent=2))


if __name__ == "__main__":
    main()
