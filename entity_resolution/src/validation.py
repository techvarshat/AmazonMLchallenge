from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def validate_matching_file(path: str | Path, test_source1_ids: Iterable[str]) -> None:
    required = set(test_source1_ids)
    seen = set()
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "source1_entity_id\tmatched_entity_ids":
            raise ValueError(f"Unexpected header in matching file: {header!r}")
        for line in f:
            if not line.strip():
                continue
            s1, matched = line.rstrip("\n").split("\t", 1)
            if s1 in seen:
                raise ValueError(f"Duplicate source1_entity_id in matching file: {s1}")
            seen.add(s1)
            if s1 not in required:
                raise ValueError(f"source1_entity_id not found in test set: {s1}")
            for mid in matched.split(",") if matched else []:
                if mid and not mid.startswith(("S2-", "S3-")):
                    raise ValueError(f"Invalid matched entity id {mid!r} for {s1}")
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Missing required source1_entity_id rows: {missing[:10]}")


def validate_candidate_file(path: str | Path, test_source1_ids: Iterable[str]) -> None:
    required = set(test_source1_ids)
    seen = set()
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "source1_entity_id\tcandidate_entity_ids":
            raise ValueError(f"Unexpected header in candidate file: {header!r}")
        for line in f:
            if not line.strip():
                continue
            s1, candidates = line.rstrip("\n").split("\t", 1)
            if s1 in seen:
                raise ValueError(f"Duplicate source1_entity_id in candidate file: {s1}")
            seen.add(s1)
            for cand in candidates.split(",") if candidates else []:
                if cand and not cand.startswith(("S2-", "S3-")):
                    raise ValueError(f"Invalid candidate entity id {cand!r} for {s1}")
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Missing required source1_entity_id rows in candidates: {missing[:10]}")
