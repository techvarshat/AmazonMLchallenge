from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd


def _tsv_dtype() -> dict[str, str]:
    return {"entity_id": "string"}


def iter_tsv_chunks(
    path: str | Path,
    chunk_size: int = 100_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    path = Path(path)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        dtype=_tsv_dtype(),
        chunksize=max(1, int(chunk_size)),
        usecols=usecols,
    ):
        yield chunk


def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a TSV file while keeping the process robust to the hackathon schema."""
    return pd.read_csv(
        path,
        sep="\t",
        engine="python",
        on_bad_lines="skip",
        dtype=_tsv_dtype(),
        nrows=nrows,
    )


def load_source_table(
    path: str | Path,
    nrows: int | None = None,
    selected_entity_ids: Iterable[str] | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    if selected_entity_ids is not None:
        selected = {str(entity_id) for entity_id in selected_entity_ids if str(entity_id).strip()}
        if not selected:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        frames: list[pd.DataFrame] = []
        for chunk in iter_tsv_chunks(Path(path), chunk_size=chunk_size):
            if "entity_id" not in chunk.columns:
                continue
            filtered = chunk[chunk["entity_id"].astype(str).isin(selected)].copy()
            if not filtered.empty:
                frames.append(filtered)
        if not frames:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        df = pd.concat(frames, ignore_index=True)
    else:
        df = read_tsv(path, nrows=nrows)

    required = {"entity_id", "business_name", "business_address", "country"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def load_ground_truth(
    path: str | Path,
    selected_source1_ids: Iterable[str] | None = None,
    limit: int | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    if selected_source1_ids is not None:
        selected = {str(entity_id) for entity_id in selected_source1_ids if str(entity_id).strip()}
        if not selected:
            return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

        frames: list[pd.DataFrame] = []
        for chunk in iter_tsv_chunks(Path(path), chunk_size=chunk_size):
            if "source1_entity_id" not in chunk.columns:
                continue
            filtered = chunk[chunk["source1_entity_id"].astype(str).isin(selected)].copy()
            if not filtered.empty:
                frames.append(filtered)
        if not frames:
            return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
        df = pd.concat(frames, ignore_index=True)
    else:
        df = read_tsv(path)
        if limit is not None:
            ordered_ids = list(dict.fromkeys(df["source1_entity_id"].astype(str).tolist()))[:limit]
            df = df[df["source1_entity_id"].astype(str).isin(ordered_ids)].copy()

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
