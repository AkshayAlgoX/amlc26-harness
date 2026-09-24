"""End-to-end Entity Resolution pipeline skeleton for AMLC26.

Integrates loading, normalization, candidate blocking, pairwise feature extraction,
matching, singleton handling, TSV file generation, and validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .blocking import MultiPassBlocker, evaluate_blocking
from .features import PairwiseFeatureExtractor
from .generation import generate_submission_package
from .loader import ERDataset, load_train_dataset, load_test_dataset
from .metric import evaluate_matching, macro_f05
from .normalize import normalize_dataframe
from .singletons import SingletonPostProcessor
from .validator import ValidationResult, validate_submission


class ERPipeline:
    """Full-pipeline skeleton for AMLC26 Business Entity Resolution."""

    def __init__(
        self,
        blocker: Optional[MultiPassBlocker] = None,
        feature_extractor: Optional[PairwiseFeatureExtractor] = None,
        post_processor: Optional[SingletonPostProcessor] = None,
    ):
        self.blocker = blocker or MultiPassBlocker()
        self.feature_extractor = feature_extractor or PairwiseFeatureExtractor()
        self.post_processor = post_processor or SingletonPostProcessor(
            match_threshold=0.5, singleton_threshold=0.5
        )

    def prepare_data(self, dataset: ERDataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize S1 records and combined S2/S3 target pool."""
        norm_s1 = normalize_dataframe(dataset.s1)
        targets_raw = dataset.target_records
        norm_targets = normalize_dataframe(targets_raw)
        return norm_s1, norm_targets

    def run_blocking(
        self, norm_s1: pd.DataFrame, norm_targets: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """Index target records and generate candidate pairs for all S1 entities."""
        self.blocker.fit(norm_targets)
        return self.blocker.block_all(norm_s1)

    def extract_features(
        self,
        candidate_map: Dict[str, Set[str]],
        norm_s1: pd.DataFrame,
        norm_targets: pd.DataFrame,
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ) -> pd.DataFrame:
        """Extract pairwise features for all candidate pairs."""
        return self.feature_extractor.extract_pairs_dataframe(
            candidate_map=candidate_map,
            s1_df=norm_s1,
            targets_df=norm_targets,
            ground_truth=ground_truth,
        )

    def score_heuristic_baseline(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute a transparent heuristic similarity score without external models.

        Combines name similarities and address similarities.
        """
        df = pairs_df.copy()
        if df.empty:
            df["score"] = 0.0
            return df

        # Weighted combination of core string similarity signals
        name_sim = (
            0.35 * df["name_token_jaccard"]
            + 0.35 * df["name_char_3gram_jaccard"]
            + 0.30 * df["name_difflib_ratio"]
        )
        addr_sim = (
            0.50 * df["addr_token_jaccard"]
            + 0.50 * df["addr_difflib_ratio"]
        )
        country_penalty = df["country_exact_match"].apply(lambda m: 1.0 if m == 1.0 else 0.7)

        # Baseline combined score
        df["score"] = (0.75 * name_sim + 0.25 * addr_sim) * country_penalty
        return df

    def predict(
        self,
        scores_df: pd.DataFrame,
        all_s1_ids: Iterable[str],
        score_col: str = "score",
    ) -> Dict[str, Set[str]]:
        """Apply singleton confidence gating and thresholding to produce final matches."""
        return self.post_processor.process(scores_df, all_s1_ids=all_s1_ids, score_col=score_col)

    def generate_outputs(
        self,
        matching_map: Dict[str, Iterable[str]],
        candidate_map: Dict[str, Iterable[str]],
        all_s1_ids: Iterable[str],
        output_dir: str | Path,
    ) -> Tuple[Path, Path]:
        """Write candidate_pairs.tsv and matching_results.tsv to output_dir."""
        return generate_submission_package(
            matching_map=matching_map,
            candidate_map=candidate_map,
            all_s1_ids=all_s1_ids,
            output_dir=output_dir,
        )

    def validate_outputs(
        self,
        matching_path: str | Path,
        candidate_path: str | Path,
        test_dir: Optional[str | Path] = None,
        required_s1_ids: Optional[Set[str]] = None,
    ) -> ValidationResult:
        """Validate generated TSV outputs with official validation rules."""
        return validate_submission(
            matching_path=matching_path,
            candidate_path=candidate_path,
            test_dir=test_dir,
            required_s1_ids=required_s1_ids,
        )
