"""Multi-pass candidate blocking for AMLC26 Business Entity Resolution.

Generates candidate pairs (S1, S2/S3) with high recall ceiling while reducing
the quadratic search space. Supports multiple composable passes.
"""
from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BlockingPass(ABC):
    """Abstract base class for a single blocking pass."""

    @abstractmethod
    def fit(self, targets_df: pd.DataFrame) -> BlockingPass:
        """Index the target records (S2 + S3)."""
        pass

    @abstractmethod
    def get_candidates(self, s1_row: pd.Series) -> Set[str]:
        """Return a set of candidate target IDs for a single S1 record."""
        pass


class ExactNameBlockingPass(BlockingPass):
    """Blocks on exact match of normalized name or core name, partitioned by country."""

    def __init__(self, use_core: bool = True, match_country: bool = True):
        self.use_core = use_core
        self.match_country = match_country
        self.index: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
        self.index_any_country: Dict[str, Set[str]] = collections.defaultdict(set)

    def _key(self, name: str, country: str) -> Tuple[str, str]:
        return (name.strip(), country.strip())

    def fit(self, targets_df: pd.DataFrame) -> ExactNameBlockingPass:
        self.index.clear()
        self.index_any_country.clear()

        name_col = "core_business_name" if self.use_core and "core_business_name" in targets_df else "norm_business_name"
        country_col = "norm_country" if "norm_country" in targets_df else "country"

        for _, row in targets_df.iterrows():
            tid = str(row["entity_id"]).strip()
            name = str(row.get(name_col, "")).strip()
            country = str(row.get(country_col, "")).strip()
            if name:
                self.index[self._key(name, country)].add(tid)
                self.index_any_country[name].add(tid)
        return self

    def get_candidates(self, s1_row: pd.Series) -> Set[str]:
        name_col = "core_business_name" if self.use_core and "core_business_name" in s1_row else "norm_business_name"
        country_col = "norm_country" if "norm_country" in s1_row else "country"

        name = str(s1_row.get(name_col, "")).strip()
        country = str(s1_row.get(country_col, "")).strip()

        if not name:
            return set()

        if self.match_country and country:
            res = self.index.get(self._key(name, country), set())
            if res:
                return set(res)
        return set(self.index_any_country.get(name, set()))


class TokenInvertedIndexPass(BlockingPass):
    """Blocks on shared distinctive business name tokens."""

    def __init__(self, min_token_len: int = 3, match_country: bool = True, max_bucket_size: int = 200):
        self.min_token_len = min_token_len
        self.match_country = match_country
        self.max_bucket_size = max_bucket_size
        self.token_to_ids: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
        self.token_to_ids_any: Dict[str, Set[str]] = collections.defaultdict(set)

    def fit(self, targets_df: pd.DataFrame) -> TokenInvertedIndexPass:
        self.token_to_ids.clear()
        self.token_to_ids_any.clear()

        tokens_col = "core_tokens" if "core_tokens" in targets_df else "name_tokens"
        country_col = "norm_country" if "norm_country" in targets_df else "country"

        for _, row in targets_df.iterrows():
            tid = str(row["entity_id"]).strip()
            tokens = row.get(tokens_col, [])
            country = str(row.get(country_col, "")).strip()
            for t in tokens:
                if len(t) >= self.min_token_len:
                    self.token_to_ids[(t, country)].add(tid)
                    self.token_to_ids_any[t].add(tid)
        return self

    def get_candidates(self, s1_row: pd.Series) -> Set[str]:
        tokens_col = "core_tokens" if "core_tokens" in s1_row else "name_tokens"
        country_col = "norm_country" if "norm_country" in s1_row else "country"

        tokens = s1_row.get(tokens_col, [])
        country = str(s1_row.get(country_col, "")).strip()

        candidates: Set[str] = set()
        for t in tokens:
            if len(t) < self.min_token_len:
                continue
            bucket = set()
            if self.match_country and country:
                bucket = self.token_to_ids.get((t, country), set())
            if not bucket:
                bucket = self.token_to_ids_any.get(t, set())

            # Skip overpopulated stop-word like tokens to avoid bloat
            if 0 < len(bucket) <= self.max_bucket_size:
                candidates.update(bucket)

        return candidates


class TfidfCosineBlockingPass(BlockingPass):
    """Cosine similarity blocking using TF-IDF on business name + address."""

    def __init__(self, top_k: int = 10, min_similarity: float = 0.25, ngram_range: Tuple[int, int] = (1, 2)):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=1, max_df=0.8, lowercase=True)
        self.target_ids: List[str] = []
        self.target_matrix = None

    def _compose_text(self, df: pd.DataFrame) -> List[str]:
        names = df.get("norm_business_name", df.get("business_name", "")).fillna("").astype(str)
        addresses = df.get("norm_business_address", df.get("business_address", "")).fillna("").astype(str)
        return (names + " " + addresses).tolist()

    def fit(self, targets_df: pd.DataFrame) -> TfidfCosineBlockingPass:
        self.target_ids = targets_df["entity_id"].astype(str).tolist()
        texts = self._compose_text(targets_df)
        self.target_matrix = self.vectorizer.fit_transform(texts)
        return self

    def get_candidates(self, s1_row: pd.Series) -> Set[str]:
        if self.target_matrix is None or not self.target_ids:
            return set()

        name = str(s1_row.get("norm_business_name", s1_row.get("business_name", "")))
        addr = str(s1_row.get("norm_business_address", s1_row.get("business_address", "")))
        text = f"{name} {addr}".strip()
        if not text:
            return set()

        query_vec = self.vectorizer.transform([text])
        sims = (query_vec * self.target_matrix.T).toarray().ravel()

        if len(sims) == 0:
            return set()

        # Find top k matches with similarity >= min_similarity
        top_indices = np.argpartition(sims, -min(self.top_k, len(sims)))[-self.top_k :]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        candidates: Set[str] = set()
        for idx in top_indices:
            if sims[idx] >= self.min_similarity:
                candidates.add(self.target_ids[idx])
        return candidates


class MultiPassBlocker:
    """Orchestrates multiple blocking passes to generate candidate pairs."""

    def __init__(
        self,
        passes: Optional[List[BlockingPass]] = None,
        max_candidates_per_s1: Optional[int] = 100,
    ):
        self.passes = passes or [
            ExactNameBlockingPass(use_core=True, match_country=True),
            ExactNameBlockingPass(use_core=False, match_country=False),
            TokenInvertedIndexPass(min_token_len=3, match_country=True, max_bucket_size=150),
            TfidfCosineBlockingPass(top_k=15, min_similarity=0.3),
        ]
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.valid_target_ids: Set[str] = set()

    def fit(self, targets_df: pd.DataFrame) -> MultiPassBlocker:
        """Fit all blocking passes on target records (S2 + S3)."""
        self.valid_target_ids = set(targets_df["entity_id"].astype(str))
        for bp in self.passes:
            bp.fit(targets_df)
        return self

    def block_entity(self, s1_row: pd.Series) -> Set[str]:
        """Generate deduplicated candidates for an S1 entity across all passes."""
        candidates: Set[str] = set()
        for bp in self.passes:
            cands = bp.get_candidates(s1_row)
            candidates.update(cands)

        # Enforce valid targets only (must exist in targets and not be self-match)
        valid = {c for c in candidates if c in self.valid_target_ids and not c.startswith("S1-")}

        if self.max_candidates_per_s1 and len(valid) > self.max_candidates_per_s1:
            valid = set(list(valid)[: self.max_candidates_per_s1])

        return valid

    def block_all(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Generate candidate set for all S1 entities in a DataFrame."""
        candidate_map: Dict[str, Set[str]] = {}
        for _, row in s1_df.iterrows():
            s1_id = str(row["entity_id"]).strip()
            candidate_map[s1_id] = self.block_entity(row)
        return candidate_map


def evaluate_blocking(
    candidate_map: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    total_targets_count: int,
) -> Dict[str, float]:
    """Calculate blocking quality metrics: recall ceiling, reduction ratio, etc."""
    total_s1 = len(ground_truth)
    if total_s1 == 0:
        return {"recall": 0.0, "reduction_ratio": 1.0, "avg_candidates": 0.0}

    total_true_matches = 0
    captured_matches = 0
    total_candidates = 0
    singletons_count = 0
    singletons_empty = 0

    for s1_id, true_set in ground_truth.items():
        cands = candidate_map.get(s1_id, set())
        total_candidates += len(cands)

        if not true_set:
            singletons_count += 1
            if not cands:
                singletons_empty += 1
        else:
            total_true_matches += len(true_set)
            captured_matches += len(true_set.intersection(cands))

    recall = captured_matches / total_true_matches if total_true_matches > 0 else 1.0
    possible_pairs = max(1, total_s1 * total_targets_count)
    reduction_ratio = 1.0 - (total_candidates / possible_pairs)
    avg_cands = total_candidates / total_s1 if total_s1 > 0 else 0.0
    singleton_empty_ratio = singletons_empty / singletons_count if singletons_count > 0 else 1.0

    return {
        "recall": float(recall),
        "captured_matches": float(captured_matches),
        "total_true_matches": float(total_true_matches),
        "total_candidates": float(total_candidates),
        "avg_candidates_per_s1": float(avg_cands),
        "reduction_ratio": float(reduction_ratio),
        "singletons_empty_ratio": float(singleton_empty_ratio),
    }
