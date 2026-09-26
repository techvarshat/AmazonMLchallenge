from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a TSV file while keeping the process robust to the hackathon schema."""
    return pd.read_csv(
        path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        dtype={"entity_id": "string"},
        nrows=nrows,
    )


def load_source_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = read_tsv(path, nrows=nrows)
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    df = read_tsv(path)
    expected = {"source1_entity_id", "matched_entity_ids"}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Ground truth file missing columns: {missing}")
    return df


def iter_source_tables(data_dir: str | Path) -> Iterable[tuple[str, pd.DataFrame]]:
    data_dir = Path(data_dir)
    for file_name in sorted(data_dir.glob("*_source*.tsv")):
        yield file_name.name, load_source_table(file_name)


def list_train_files(train_dir: str | Path) -> List[Path]:
    train_dir = Path(train_dir)
    return sorted(train_dir.glob("*.tsv"))
