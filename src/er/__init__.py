"""AMLC26 Business Entity Resolution package."""
from .blocking import (
    BlockingPass,
    ExactNameBlockingPass,
    MultiPassBlocker,
    TfidfCosineBlockingPass,
    TokenInvertedIndexPass,
    evaluate_blocking,
)
from .features import PairwiseFeatureExtractor
from .ids import code_source, decode_target_id, encode_target_id, is_target_id, source_of
from .generation import (
    generate_candidate_pairs_tsv,
    generate_matching_results_tsv,
    generate_submission_package,
)
from .loader import (
    ERDataset,
    format_id_list,
    load_all_7_tsvs,
    load_ground_truth,
    load_source_tsv,
    load_test_dataset,
    load_train_dataset,
    parse_id_list,
)
from .metric import evaluate_matching, f05_score_single, macro_f05
from .normalize import (
    clean_text,
    extract_name_core,
    extract_numbers,
    normalize_address,
    normalize_country,
    normalize_dataframe,
    normalize_name,
)
from .pipeline import ERPipeline
from .singletons import SingletonPostProcessor, tune_singleton_thresholds
from .split import (
    filter_candidate_map_by_s1,
    filter_ground_truth_by_s1,
    filter_pairs_df_by_s1,
    s1_kfold_split,
    s1_train_val_split,
)
from .validator import ValidationResult, validate_submission

__all__ = [
    "source_of",
    "is_target_id",
    "encode_target_id",
    "decode_target_id",
    "code_source",
    "load_source_tsv",
    "load_ground_truth",
    "load_train_dataset",
    "load_test_dataset",
    "load_all_7_tsvs",
    "parse_id_list",
    "format_id_list",
    "ERDataset",
    "clean_text",
    "normalize_name",
    "extract_name_core",
    "normalize_address",
    "normalize_country",
    "extract_numbers",
    "normalize_dataframe",
    "BlockingPass",
    "ExactNameBlockingPass",
    "TokenInvertedIndexPass",
    "TfidfCosineBlockingPass",
    "MultiPassBlocker",
    "evaluate_blocking",
    "PairwiseFeatureExtractor",
    "s1_train_val_split",
    "s1_kfold_split",
    "filter_candidate_map_by_s1",
    "filter_ground_truth_by_s1",
    "filter_pairs_df_by_s1",
    "f05_score_single",
    "macro_f05",
    "evaluate_matching",
    "SingletonPostProcessor",
    "tune_singleton_thresholds",
    "generate_candidate_pairs_tsv",
    "generate_matching_results_tsv",
    "generate_submission_package",
    "validate_submission",
    "ValidationResult",
    "ERPipeline",
]
