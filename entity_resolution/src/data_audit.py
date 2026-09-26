from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .data_loader import load_source_table


@dataclass
class DatasetAudit:
    file_name: str
    row_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_pct: dict[str, float]
    unique_count: dict[str, int]
    duplicate_rate: float


def audit_dataset(file_path: str | Path, nrows: int | None = None) -> DatasetAudit:
    df = load_source_table(file_path, nrows=nrows)
    missing_pct = {col: round(float(df[col].isna().mean() * 100), 3) for col in df.columns}
    unique_count = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
    duplicate_rate = float(df.duplicated(subset=["entity_id"]).mean() * 100) if "entity_id" in df.columns else 0.0
    return DatasetAudit(
        file_name=Path(file_path).name,
        row_count=int(len(df)),
        columns=list(df.columns),
        dtypes={str(k): str(v) for k, v in df.dtypes.items()},
        missing_pct=missing_pct,
        unique_count=unique_count,
        duplicate_rate=duplicate_rate,
    )


def summarize_dataset(data_dir: str | Path, nrows: int | None = None) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    audits = {}
    for file_path in sorted(data_dir.glob("*.tsv")):
        if not file_path.name.endswith(("_source1.tsv", "_source2.tsv", "_source3.tsv")):
            continue
        audits[file_path.name] = audit_dataset(file_path, nrows=nrows)
    return audits


def print_audit_report(audit_map: Dict[str, DatasetAudit]) -> None:
    for name, audit in sorted(audit_map.items()):
        print(f"\n=== {name} ===")
        print(f"rows={audit.row_count}")
        print(f"columns={audit.columns}")
        print(f"duplicate_entity_id_rate_pct={audit.duplicate_rate}")
        print(f"missing_pct={audit.missing_pct}")
        print(f"unique_counts={audit.unique_count}")
