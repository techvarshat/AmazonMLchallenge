from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def write_candidate_pairs(path: str | Path, candidates: Dict[str, List[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for source1_id in sorted(candidates):
            ids = sorted(set(candidates[source1_id]))
            f.write(f"{source1_id}\t{','.join(ids)}\n")


def write_matching_results(path: str | Path, matches: Dict[str, List[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for source1_id in sorted(matches):
            ids = sorted(set(matches[source1_id]))
            f.write(f"{source1_id}\t{','.join(ids)}\n")
