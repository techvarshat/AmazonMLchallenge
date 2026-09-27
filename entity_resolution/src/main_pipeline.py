from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .blocking import generate_exact_candidates, generate_token_candidates, merge_candidate_sets
from .data_audit import print_audit_report, summarize_dataset
from .data_loader import load_ground_truth, load_source_table
from .evaluation import macro_f05_score
from .normalization import compact_text, normalize_text, strip_business_suffixes, token_list
from .pair_features import build_pair_features
from .threshold_optimization import optimize_threshold
from .train_lgbm import train_lightgbm_model


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].fillna("").map(normalize_text)
    out["address_norm"] = out["business_address"].fillna("").map(normalize_text)
    out["name_compact"] = out["business_name"].fillna("").map(compact_text)
    out["name_tokens"] = out["name_norm"].map(token_list)
    out["tokens"] = out.apply(lambda r: token_list(r["business_name"]) + token_list(r["business_address"]), axis=1)
    out["country_norm"] = out["country"].fillna("").map(lambda x: normalize_text(str(x)))
    out["name_no_suffix"] = out["business_name"].fillna("").map(strip_business_suffixes)
    return out


def _build_truth_map(truth_df: pd.DataFrame) -> dict[str, set[str]]:
    truth_map: dict[str, set[str]] = {}
    for row in truth_df.itertuples(index=False):
        s1 = str(row.source1_entity_id)
        matched = row.matched_entity_ids
        truth_map[s1] = set() if pd.isna(matched) or str(matched).strip() == "" else {x.strip() for x in str(matched).split(",") if x.strip()}
    return truth_map


def _generate_candidate_pairs(s1_df: pd.DataFrame, target_df: pd.DataFrame, max_candidates: int = 50) -> dict[str, set[str]]:
    exact = generate_exact_candidates(s1_df.to_dict("records"), target_df.to_dict("records"), [])
    token = generate_token_candidates(s1_df.to_dict("records"), target_df.to_dict("records"), [], max_candidates=max_candidates)
    merged = merge_candidate_sets(exact, token)
    for entity_id in list(merged):
        if len(merged[entity_id]) > max_candidates:
            merged[entity_id] = set(list(merged[entity_id])[:max_candidates])
    return merged


def _feature_row(left: dict, right: dict) -> dict:
    features = build_pair_features(left, right)
    return features


def _build_training_rows(source1_df: pd.DataFrame, target_df: pd.DataFrame, truth_map: dict[str, set[str]], max_candidates: int = 50) -> tuple[list[dict], list[int]]:
    candidate_map = _generate_candidate_pairs(source1_df, target_df, max_candidates=max_candidates)
    rows: list[dict] = []
    labels: list[int] = []
    for s1_id, row in source1_df.set_index("entity_id").to_dict("index").items():
        pass

    for s1_id, row in source1_df.set_index("entity_id").iterrows():
        left = row.to_dict()
        left["tokens"] = row["tokens"] if "tokens" in row else []
        left["name_norm"] = row["name_norm"]
        left["name_compact"] = row["name_compact"]
        left["address_norm"] = row["address_norm"]
        left["name_tokens"] = row["name_tokens"]
        candidates = candidate_map.get(str(s1_id), set())
        for cand_id in candidates:
            right = target_df[target_df["entity_id"] == cand_id].to_dict("records")
            if not right:
                continue
            right = right[0]
            right["tokens"] = right.get("tokens", [])
            right["name_norm"] = right.get("name_norm", "")
            right["name_compact"] = right.get("name_compact", "")
            right["address_norm"] = right.get("address_norm", "")
            right["name_tokens"] = right.get("name_tokens", [])
            features = _feature_row(left, right)
            rows.append(features)
            labels.append(int(cand_id in truth_map.get(str(s1_id), set())))

    if not rows:
        raise ValueError("No rows generated. Check candidate generation or target set.")
    return rows, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entity resolution pipeline for the business entity match challenge.")
    parser.add_argument("--data-root", type=str, default="../6ab10eb3b23ba_student_resource/student_resource")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for quick smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print("[1/6] Loading data")
    train_dir = data_root / "dataset" / "train"
    full_truth = load_ground_truth(train_dir / "train_ground_truth.tsv")
    if args.limit is not None:
        limited_truth = full_truth.head(min(args.limit, len(full_truth))).copy()
        selected_s1_ids = limited_truth["source1_entity_id"].drop_duplicates().astype(str).tolist()
        selected_target_ids = set()
        for matched in limited_truth["matched_entity_ids"]:
            if pd.isna(matched) or not str(matched).strip():
                continue
            selected_target_ids.update({x.strip() for x in str(matched).split(",") if x.strip()})
        s1 = _normalize_frame(load_source_table(train_dir / "train_source1.tsv"))
        s2 = _normalize_frame(load_source_table(train_dir / "train_source2.tsv"))
        s3 = _normalize_frame(load_source_table(train_dir / "train_source3.tsv"))
        s1 = s1[s1["entity_id"].astype(str).isin(selected_s1_ids)].copy()
        target_ids = set(pd.concat([s2, s3], ignore_index=True)["entity_id"].astype(str).tolist())
        sampled_target_ids = selected_target_ids & target_ids
        if not sampled_target_ids:
            raise ValueError(
                "The --limit subset does not contain any valid ground-truth target IDs in the sampled target tables; this creates an all-negative label set. "
                "Use a larger limit or remove --limit to keep truth and target data aligned."
            )
        s2 = s2[s2["entity_id"].astype(str).isin(sampled_target_ids | set(list(target_ids - selected_target_ids)[:50]))].copy()
        s3 = s3[s3["entity_id"].astype(str).isin(sampled_target_ids | set(list(target_ids - selected_target_ids)[:50]))].copy()
        truth = limited_truth[limited_truth["source1_entity_id"].astype(str).isin(selected_s1_ids)].copy()
    else:
        s1 = _normalize_frame(load_source_table(train_dir / "train_source1.tsv"))
        s2 = _normalize_frame(load_source_table(train_dir / "train_source2.tsv"))
        s3 = _normalize_frame(load_source_table(train_dir / "train_source3.tsv"))
        truth = full_truth
    truth_map = _build_truth_map(truth)
    print(f"Loaded S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, truth={len(truth_map)}")

    print("[2/6] Auditing data")
    audit_map = summarize_dataset(train_dir, nrows=args.limit)
    print_audit_report(audit_map)

    print("[3/6] Building candidate pairs")
    target = pd.concat([s2, s3], ignore_index=True)
    candidate_map = _generate_candidate_pairs(s1, target, max_candidates=25)
    print(f"Generated candidates for {len(candidate_map)} S1 entities; median candidates={np.median([len(v) for v in candidate_map.values()]) if candidate_map else 0}")

    print("[4/6] Training LightGBM-only model")
    rows, labels = _build_training_rows(s1, target, truth_map, max_candidates=25)
    feature_frame = pd.DataFrame(rows)
    feature_frame = feature_frame.fillna(0)
    train_idx, val_idx = train_test_split(np.arange(len(feature_frame)), test_size=0.2, random_state=42, stratify=np.asarray(labels))
    X_train = feature_frame.iloc[train_idx].copy()
    X_val = feature_frame.iloc[val_idx].copy()
    y_train = np.asarray([labels[i] for i in train_idx])
    y_val = np.asarray([labels[i] for i in val_idx])

    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        print("Warning: reduced smoke-test sample does not contain both classes; using default threshold 0.5.")
        val_metrics = 0.0
        best_threshold = 0.5
    else:
        lgb_model = train_lightgbm_model(X_train.to_dict("records"), y_train)
        lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
        best_threshold, best_f05 = optimize_threshold(lgb_prob, y_val, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
        val_metrics = best_f05
        print(f"Validation F0.5 = {val_metrics:.4f} using threshold={best_threshold:.2f}")

    print("[5/6] Ranking check")
    best_candidate = []
    for s1_id in s1["entity_id"].tolist()[:5]:
        if s1_id not in candidate_map:
            continue
        best_candidate.append((s1_id, sorted(candidate_map[s1_id], key=lambda x: random.random())[:3]))
    print(f"Example candidates: {best_candidate[:2]}")

    print("[6/6] Submission package")
    print(f"Outputs directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
