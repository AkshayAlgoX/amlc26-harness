"""Train/validation splitting for Business Entity Resolution.

Strictly groups by Source 1 entity IDs so that all candidate pairs and
ground truth labels for a given S1 entity belong exclusively to either
the train split or the validation split. Zero leakage across splits.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def make_strata(s1_df: pd.DataFrame, ground_truth: Optional[Dict[str, Set[str]]] = None) -> pd.Series:
    """Create a stratification key combining country and singleton status."""
    countries = s1_df.get("norm_country", s1_df.get("country", "")).fillna("UNKNOWN").astype(str)

    if ground_truth is not None:
        is_singleton = s1_df["entity_id"].apply(lambda sid: 1 if len(ground_truth.get(str(sid), set())) == 0 else 0)
        strata = countries + "_" + is_singleton.astype(str)
    else:
        strata = countries

    return strata


def s1_train_val_split(
    s1_df: pd.DataFrame,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[str], List[str]]:
    """Split S1 entities into train and validation sets with zero S1 overlap.

    Returns: (train_s1_ids, val_s1_ids)
    """
    n = len(s1_df)
    if n == 0:
        return [], []

    s1_ids = s1_df["entity_id"].astype(str).tolist()

    if not stratify or n < 10:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(s1_ids).tolist()
        split_idx = int(n * (1.0 - val_size))
        return shuffled[:split_idx], shuffled[split_idx:]

    strata = make_strata(s1_df, ground_truth)
    n_splits = max(2, int(round(1.0 / val_size)))
    counts = strata.value_counts()
    if (counts < n_splits).any():
        # Fallback to random KFold if rare stratum prevents StratifiedKFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(kf.split(s1_ids))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(skf.split(s1_ids, strata))

    train_ids = [s1_ids[i] for i in train_idx]
    val_ids = [s1_ids[i] for i in val_idx]
    return train_ids, val_ids


def s1_kfold_split(
    s1_df: pd.DataFrame,
    ground_truth: Optional[Dict[str, Set[str]]] = None,
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
) -> List[Tuple[List[str], List[str]]]:
    """Generate K train/val splits of S1 entity IDs with zero leakage across folds.

    Returns list of (train_s1_ids, val_s1_ids) tuples for each fold.
    """
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    n = len(s1_ids)

    if stratify and n >= n_splits:
        strata = make_strata(s1_df, ground_truth)
        counts = strata.value_counts()
        if (counts >= n_splits).all():
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = []
            for tr_idx, val_idx in skf.split(s1_ids, strata):
                splits.append(([s1_ids[i] for i in tr_idx], [s1_ids[i] for i in val_idx]))
            return splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for tr_idx, val_idx in kf.split(s1_ids):
        splits.append(([s1_ids[i] for i in tr_idx], [s1_ids[i] for i in val_idx]))
    return splits


def filter_candidate_map_by_s1(
    candidate_map: Dict[str, Set[str]], s1_ids: Iterable[str]
) -> Dict[str, Set[str]]:
    """Filter a candidate map to only include the specified S1 entity IDs."""
    s1_set = set(str(s).strip() for s in s1_ids)
    return {sid: cands for sid, cands in candidate_map.items() if sid in s1_set}


def filter_ground_truth_by_s1(
    ground_truth: Dict[str, Set[str]], s1_ids: Iterable[str]
) -> Dict[str, Set[str]]:
    """Filter ground truth dictionary to only include the specified S1 entity IDs."""
    s1_set = set(str(s).strip() for s in s1_ids)
    return {sid: trues for sid, trues in ground_truth.items() if sid in s1_set}


def filter_pairs_df_by_s1(pairs_df: pd.DataFrame, s1_ids: Iterable[str]) -> pd.DataFrame:
    """Filter a pairwise feature DataFrame to only include specified S1 entity IDs."""
    s1_set = set(str(s).strip() for s in s1_ids)
    return pairs_df[pairs_df["source1_entity_id"].isin(s1_set)].copy()
