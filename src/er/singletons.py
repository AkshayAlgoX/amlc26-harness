"""Singleton handling module for AMLC26 Business Entity Resolution.

In F_0.5, a singleton (0 true matches) scores 1.0 when predicted empty,
and 0.0 if ANY match is predicted. This module provides confidence gating,
thresholding, and threshold tuning to guard against false merges on singletons.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .metric import macro_f05


class SingletonPostProcessor:
    """Post-processor that converts pairwise scores into discrete entity matches

    with explicit singleton gating.
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        singleton_threshold: float = 0.5,
        max_matches_per_s1: Optional[int] = 10,
    ):
        """Args:

        match_threshold: Minimum pair score to consider candidate a match.
        singleton_threshold: If max candidate score for an S1 entity is below this,
                             the S1 entity is forced to be a singleton (empty match set).
        max_matches_per_s1: Maximum matches to keep per S1 entity.
        """
        self.match_threshold = match_threshold
        self.singleton_threshold = singleton_threshold
        self.max_matches_per_s1 = max_matches_per_s1

    def process(
        self,
        scores_df: pd.DataFrame,
        all_s1_ids: Iterable[str],
        score_col: str = "score",
    ) -> Dict[str, Set[str]]:
        """Convert a pairwise scores DataFrame into a {s1_id: set_of_matched_ids} dict.

        scores_df must contain: 'source1_entity_id', 'target_entity_id', and score_col.
        all_s1_ids guarantees every S1 entity is present (even if no candidates).
        """
        all_s1_set = [str(s).strip() for s in all_s1_ids]
        matches: Dict[str, Set[str]] = {sid: set() for sid in all_s1_set}

        if scores_df.empty:
            return matches

        grouped = scores_df.groupby("source1_entity_id")

        for s1_id, group in grouped:
            s1_id_str = str(s1_id).strip()
            if s1_id_str not in matches:
                continue

            max_score = group[score_col].max()
            # Singleton gate: if best score is below singleton_threshold, abstain completely
            if max_score < self.singleton_threshold:
                matches[s1_id_str] = set()
                continue

            # Filter candidates above match_threshold
            qualifying = group[group[score_col] >= self.match_threshold]
            if qualifying.empty:
                matches[s1_id_str] = set()
                continue

            # Sort descending by score
            qualifying = qualifying.sort_values(by=score_col, ascending=False)
            if self.max_matches_per_s1:
                qualifying = qualifying.iloc[: self.max_matches_per_s1]

            matched_ids = set(qualifying["target_entity_id"].astype(str).tolist())
            matches[s1_id_str] = matched_ids

        return matches


def tune_singleton_thresholds(
    scores_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Iterable[str],
    score_col: str = "score",
    match_thresholds: Optional[List[float]] = None,
    singleton_thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Find the combination of (match_threshold, singleton_threshold) that maximizes Macro F0.5.

    Returns: (best_match_threshold, best_singleton_threshold, best_macro_f05)
    """
    if match_thresholds is None:
        match_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if singleton_thresholds is None:
        singleton_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_m_thresh = 0.5
    best_s_thresh = 0.5
    best_score = -1.0

    for m_th in match_thresholds:
        for s_th in singleton_thresholds:
            # Singleton threshold is usually >= match_threshold
            if s_th < m_th - 0.2:
                continue
            proc = SingletonPostProcessor(match_threshold=m_th, singleton_threshold=s_th)
            preds = proc.process(scores_df, all_s1_ids, score_col=score_col)
            sc = macro_f05(ground_truth, preds, s1_ids=all_s1_ids)

            if sc > best_score:
                best_score = sc
                best_m_thresh = m_th
                best_s_thresh = s_th

    return best_m_thresh, best_s_thresh, best_score
