"""Data loading for the 7 official TSV files of AMLC26 Business Entity Resolution.

Official files:
  Train:
    - train_source1.tsv       (deduplicated reference source)
    - train_source2.tsv
    - train_source3.tsv
    - train_ground_truth.tsv  (source1_entity_id -> matched_entity_ids)
  Test:
    - test_source1.tsv        (all S1 entities requiring predictions)
    - test_source2.tsv
    - test_source3.tsv

All files are TAB-separated (.tsv). IDs are strictly strings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def parse_id_list(val: object) -> List[str]:
    """Parse comma-separated entity IDs string into a clean list of strings."""
    if val is None or pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def format_id_list(ids: Iterable[str]) -> str:
    """Format an iterable of IDs into a canonical comma-separated string without duplicates."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in ids:
        item_str = str(item).strip()
        if item_str and item_str not in seen:
            seen.add(item_str)
            ordered.append(item_str)
    return ",".join(ordered)


def load_source_tsv(path: str | Path) -> pd.DataFrame:
    """Load an entity source TSV file (source1, source2, or source3).

    Ensures correct tab delimiter, dtypes, string stripping, and null handling.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str,
        },
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing expected column '{col}'. Found: {list(df.columns)}")
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df[SOURCE_COLUMNS]


def load_ground_truth(path: str | Path) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv as a dict mapping source1_entity_id -> set of matched_entity_ids.

    Singletons map to an empty set.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    for col in GROUND_TRUTH_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing expected column '{col}'. Found: {list(df.columns)}")

    mapping: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched = set(parse_id_list(row["matched_entity_ids"]))
        mapping[s1_id] = matched

    return mapping


def load_ground_truth_df(path: str | Path) -> pd.DataFrame:
    """Load train_ground_truth.tsv as a DataFrame with parsed list and set columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    df["source1_entity_id"] = df["source1_entity_id"].fillna("").astype(str).str.strip()
    df["matched_list"] = df["matched_entity_ids"].apply(parse_id_list)
    df["num_matches"] = df["matched_list"].apply(len)
    df["is_singleton"] = df["num_matches"] == 0
    return df


@dataclass
class ERDataset:
    """Container for Entity Resolution sources and optional ground truth."""
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    ground_truth: Optional[Dict[str, Set[str]]] = None

    @property
    def s1_ids(self) -> List[str]:
        return self.s1["entity_id"].tolist()

    @property
    def target_records(self) -> pd.DataFrame:
        """Combined S2 and S3 pool of records to match against."""
        return pd.concat([self.s2, self.s3], ignore_index=True)

    @property
    def valid_target_ids(self) -> Set[str]:
        """Set of all valid S2 and S3 entity IDs."""
        return set(self.s2["entity_id"]).union(set(self.s3["entity_id"]))


def load_train_dataset(train_dir: str | Path) -> ERDataset:
    """Load all 4 training files from a directory."""
    train_dir = Path(train_dir)
    s1 = load_source_tsv(train_dir / "train_source1.tsv")
    s2 = load_source_tsv(train_dir / "train_source2.tsv")
    s3 = load_source_tsv(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    return ERDataset(s1=s1, s2=s2, s3=s3, ground_truth=gt)


def load_test_dataset(test_dir: str | Path) -> ERDataset:
    """Load all 3 test files from a directory."""
    test_dir = Path(test_dir)
    s1 = load_source_tsv(test_dir / "test_source1.tsv")
    s2 = load_source_tsv(test_dir / "test_source2.tsv")
    s3 = load_source_tsv(test_dir / "test_source3.tsv")
    return ERDataset(s1=s1, s2=s2, s3=s3, ground_truth=None)


def load_all_7_tsvs(dataset_dir: str | Path) -> Tuple[ERDataset, ERDataset]:
    """Load all 7 official TSVs from standard dataset layout.

    Expects dataset_dir to contain 'train/' and 'test/' subdirectories.
    Returns (train_dataset, test_dataset).
    """
    root = Path(dataset_dir)
    train_dir = root / "train" if (root / "train").is_dir() else root
    test_dir = root / "test" if (root / "test").is_dir() else root

    train_data = load_train_dataset(train_dir)
    test_data = load_test_dataset(test_dir)
    return train_data, test_data
