"""Data loading. Convention: ids are ALWAYS strings everywhere in the harness."""
from __future__ import annotations

import pandas as pd

from .config import col, data_path, task, text_cols


def read_table(path, id_col: str | None = None) -> pd.DataFrame:
    id_col = id_col or col("id")
    df = pd.read_csv(path, dtype={id_col: str}, low_memory=False)
    df[id_col] = df[id_col].astype(str)
    return df


def _fix_target(df: pd.DataFrame) -> pd.DataFrame:
    tgt = col("target")
    if tgt in df.columns and task()["type"] in ("classification", "extraction"):
        df[tgt] = df[tgt].fillna("").astype(str)
    return df


def load_train() -> pd.DataFrame:
    return _fix_target(read_table(data_path("train")))


def load_test() -> pd.DataFrame:
    return read_table(data_path("test"))


def load_folds() -> pd.DataFrame:
    p = data_path("folds")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. The captain runs `python -m src.folds build` once and commits it."
        )
    f = pd.read_csv(p, dtype={"id": str})
    f["id"] = f["id"].astype(str)
    f["fold"] = f["fold"].astype(int)
    return f


def train_with_folds() -> pd.DataFrame:
    """Train rows with a `fold` column attached (fails loudly if any id has no fold)."""
    tr = load_train()
    folds = load_folds()
    out = tr.merge(folds, left_on=col("id"), right_on="id", how="left", validate="one_to_one")
    if out["fold"].isna().any():
        raise ValueError(f"{int(out['fold'].isna().sum())} train ids missing from folds.csv — rebuild folds")
    if col("id") != "id":
        out = out.drop(columns=["id"])
    out["fold"] = out["fold"].astype(int)
    return out


def text_series(df: pd.DataFrame, max_chars: int | None = None) -> pd.Series:
    """Concatenate the configured text columns into one string per row."""
    cols = [c for c in text_cols() if c in df.columns]
    if not cols:
        raise ValueError("No configured text columns found in dataframe (config: columns.text)")
    s = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        s = s + " | " + df[c].fillna("").astype(str)
    return s.str.slice(0, max_chars) if max_chars else s
