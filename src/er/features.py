"""Pairwise feature extraction for candidate pairs in Business Entity Resolution.

Extracts string similarities, token overlap metrics, address components,
and country matching features without external heavy dependencies.
"""
from __future__ import annotations

import difflib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .ids import is_target_id, source_of


def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


def containment_min(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute containment: intersection / min(len(set1), len(set2))."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    m = min(len(set1), len(set2))
    return float(intersection / m) if m > 0 else 0.0


def ngrams(text: str, n: int) -> Set[str]:
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def sequence_similarity(s1: str, s2: str) -> float:
    """Compute difflib SequenceMatcher ratio between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(difflib.SequenceMatcher(None, s1, s2).ratio())


FEATURE_NAMES: List[str] = [
    "name_exact_match",
    "name_norm_exact_match",
    "name_core_exact_match",
    "name_token_jaccard",
    "name_token_overlap_min",
    "name_token_overlap_max",
    "name_char_2gram_jaccard",
    "name_char_3gram_jaccard",
    "name_difflib_ratio",
    "name_first_token_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_token_jaccard",
    "addr_token_overlap_min",
    "addr_difflib_ratio",
    "addr_number_jaccard",
    "addr_exact_match",
    "country_exact_match",
    "country_either_missing",
    "is_source2",
    "is_source3",
    "name_x_addr_jaccard",
    "max_name_sim",
]


class PairwiseFeatureExtractor:
    """Extracts tabular pairwise features for candidate pairs (S1, S2/S3)."""

    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def extract_pair(self, s1: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, float]:
        """Extract feature dictionary for a single S1 and candidate pair."""
        feats: Dict[str, float] = {}

        # Raw / normalized strings
        raw_name1 = str(s1.get("business_name", ""))
        raw_name2 = str(cand.get("business_name", ""))
        norm_name1 = str(s1.get("norm_business_name", raw_name1))
        norm_name2 = str(cand.get("norm_business_name", raw_name2))
        core_name1 = str(s1.get("core_business_name", norm_name1))
        core_name2 = str(cand.get("core_business_name", norm_name2))

        # Tokens
        tokens1 = set(s1.get("name_tokens", norm_name1.split()))
        tokens2 = set(cand.get("name_tokens", norm_name2.split()))

        # Name exact matches
        feats["name_exact_match"] = 1.0 if raw_name1 and raw_name1 == raw_name2 else 0.0
        feats["name_norm_exact_match"] = 1.0 if norm_name1 and norm_name1 == norm_name2 else 0.0
        feats["name_core_exact_match"] = 1.0 if core_name1 and core_name1 == core_name2 else 0.0

        # Name token metrics
        t_jacc = jaccard_similarity(tokens1, tokens2)
        t_cont_min = containment_min(tokens1, tokens2)
        intersection_len = len(tokens1.intersection(tokens2))
        max_len = max(len(tokens1), len(tokens2))
        t_cont_max = float(intersection_len / max_len) if max_len > 0 else (1.0 if not tokens1 and not tokens2 else 0.0)

        feats["name_token_jaccard"] = t_jacc
        feats["name_token_overlap_min"] = t_cont_min
        feats["name_token_overlap_max"] = t_cont_max

        # Name character n-grams
        g2_1 = ngrams(norm_name1, 2)
        g2_2 = ngrams(norm_name2, 2)
        g3_1 = ngrams(norm_name1, 3)
        g3_2 = ngrams(norm_name2, 3)

        g2_jacc = jaccard_similarity(g2_1, g2_2)
        g3_jacc = jaccard_similarity(g3_1, g3_2)
        feats["name_char_2gram_jaccard"] = g2_jacc
        feats["name_char_3gram_jaccard"] = g3_jacc

        # Sequence distance
        feats["name_difflib_ratio"] = sequence_similarity(norm_name1, norm_name2)

        # First token match
        first1 = norm_name1.split()[0] if norm_name1.split() else ""
        first2 = norm_name2.split()[0] if norm_name2.split() else ""
        feats["name_first_token_match"] = 1.0 if first1 and first1 == first2 else 0.0

        # Length comparisons
        l1, l2 = len(norm_name1), len(norm_name2)
        feats["name_len_diff"] = float(abs(l1 - l2))
        feats["name_len_ratio"] = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0

        # Address features
        norm_addr1 = str(s1.get("norm_business_address", s1.get("business_address", "")))
        norm_addr2 = str(cand.get("norm_business_address", cand.get("business_address", "")))
        addr_tok1 = set(s1.get("address_tokens", norm_addr1.split()))
        addr_tok2 = set(cand.get("address_tokens", norm_addr2.split()))

        a_jacc = jaccard_similarity(addr_tok1, addr_tok2)
        feats["addr_token_jaccard"] = a_jacc
        feats["addr_token_overlap_min"] = containment_min(addr_tok1, addr_tok2)
        feats["addr_difflib_ratio"] = sequence_similarity(norm_addr1, norm_addr2)
        feats["addr_exact_match"] = 1.0 if norm_addr1 and norm_addr1 == norm_addr2 else 0.0

        num1 = set(s1.get("address_numbers", []))
        num2 = set(cand.get("address_numbers", []))
        feats["addr_number_jaccard"] = jaccard_similarity(num1, num2) if (num1 or num2) else 0.0

        # Country features
        c1 = str(s1.get("norm_country", s1.get("country", ""))).strip().lower()
        c2 = str(cand.get("norm_country", cand.get("country", ""))).strip().lower()

        feats["country_exact_match"] = 1.0 if c1 and c1 == c2 else 0.0
        feats["country_either_missing"] = 1.0 if not c1 or not c2 else 0.0

        # Target source indicator
        cand_id = str(cand.get("entity_id", ""))
        cand_src = source_of(cand_id) if is_target_id(cand_id) else ""
        feats["is_source2"] = 1.0 if cand_src == "S2" else 0.0
        feats["is_source3"] = 1.0 if cand_src == "S3" else 0.0

        # Composite features
        feats["name_x_addr_jaccard"] = t_jacc * a_jacc
        feats["max_name_sim"] = max(t_jacc, g2_jacc, g3_jacc, feats["name_difflib_ratio"])

        return feats

    def extract_pairs_dataframe(
        self,
        candidate_map: Dict[str, Set[str]],
        s1_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ) -> pd.DataFrame:
        """Extract tabular pairwise features for all candidate pairs in candidate_map.

        Returns DataFrame with ['source1_entity_id', 'target_entity_id'] + FEATURE_NAMES (+ 'label' if ground_truth).
        """
        s1_by_id = {str(row["entity_id"]).strip(): row.to_dict() for _, row in s1_df.iterrows()}
        targets_by_id = {str(row["entity_id"]).strip(): row.to_dict() for _, row in targets_df.iterrows()}

        rows = []
        for s1_id, cand_ids in candidate_map.items():
            s1_row = s1_by_id.get(s1_id)
            if not s1_row:
                continue

            true_set = ground_truth.get(s1_id, set()) if ground_truth is not None else None

            for cid in cand_ids:
                cand_row = targets_by_id.get(cid)
                if not cand_row:
                    continue

                feat_dict = self.extract_pair(s1_row, cand_row)
                feat_dict["source1_entity_id"] = s1_id
                feat_dict["target_entity_id"] = cid

                if true_set is not None:
                    feat_dict["label"] = 1 if cid in true_set else 0

                rows.append(feat_dict)

        if not rows:
            cols = ["source1_entity_id", "target_entity_id"] + self.feature_names
            if ground_truth is not None:
                cols.append("label")
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows)
        # Ensure ordered column layout
        front_cols = ["source1_entity_id", "target_entity_id"]
        tail_cols = ["label"] if "label" in df.columns else []
        df = df[front_cols + self.feature_names + tail_cols]
        return df
