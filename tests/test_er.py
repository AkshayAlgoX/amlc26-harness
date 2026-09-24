"""Comprehensive test suite for the AMLC26 Business Entity Resolution modules."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.er import (
    ERDataset,
    ERPipeline,
    ExactNameBlockingPass,
    MultiPassBlocker,
    PairwiseFeatureExtractor,
    SingletonPostProcessor,
    TfidfCosineBlockingPass,
    TokenInvertedIndexPass,
    clean_text,
    evaluate_blocking,
    evaluate_matching,
    extract_name_core,
    extract_numbers,
    f05_score_single,
    filter_candidate_map_by_s1,
    filter_ground_truth_by_s1,
    filter_pairs_df_by_s1,
    format_id_list,
    generate_candidate_pairs_tsv,
    generate_matching_results_tsv,
    generate_submission_package,
    load_all_7_tsvs,
    load_ground_truth,
    load_source_tsv,
    load_test_dataset,
    load_train_dataset,
    macro_f05,
    normalize_address,
    normalize_country,
    normalize_dataframe,
    normalize_name,
    parse_id_list,
    s1_kfold_split,
    s1_train_val_split,
    tune_singleton_thresholds,
    validate_submission,
)


@pytest.fixture
def mock_dataset_dir(tmp_path: Path):
    """Create a mock 7-file dataset layout matching the official format."""
    train_dir = tmp_path / "dataset" / "train"
    test_dir = tmp_path / "dataset" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Train files
    s1_train = pd.DataFrame(
        [
            {"entity_id": "S1-0001", "business_name": "Acme Corp Ltd.", "business_address": "123 Main St, Springfield", "country": "US"},
            {"entity_id": "S1-0002", "business_name": "Bharat Traders Pvt", "business_address": "45 MG Rd, Bangalore", "country": "India"},
            {"entity_id": "S1-0003", "business_name": "Solo Venture", "business_address": "10 Park Ave, NY", "country": "US"},  # singleton
        ]
    )
    s2_train = pd.DataFrame(
        [
            {"entity_id": "S2-0001", "business_name": "Acme Corporation", "business_address": "123 Main Street", "country": "US"},
            {"entity_id": "S2-0002", "business_name": "Bharat Traders Private Limited", "business_address": "45 M.G. Road", "country": "India"},
        ]
    )
    s3_train = pd.DataFrame(
        [
            {"entity_id": "S3-0001", "business_name": "Acme Inc.", "business_address": "123 Main St.", "country": "US"},
            {"entity_id": "S3-0002", "business_name": "Unrelated Shop", "business_address": "99 Market St", "country": "US"},
        ]
    )
    gt_train = pd.DataFrame(
        [
            {"source1_entity_id": "S1-0001", "matched_entity_ids": "S2-0001,S3-0001"},
            {"source1_entity_id": "S1-0002", "matched_entity_ids": "S2-0002"},
            {"source1_entity_id": "S1-0003", "matched_entity_ids": ""},  # singleton
        ]
    )

    s1_train.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    s2_train.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    s3_train.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    gt_train.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Test files (includes France for open-set country check)
    s1_test = pd.DataFrame(
        [
            {"entity_id": "S1-T001", "business_name": "Global Tech LLC", "business_address": "500 Tech Blvd", "country": "US"},
            {"entity_id": "S1-T002", "business_name": "Cafe De Paris SARL", "business_address": "12 Rue de Rivoli, Paris", "country": "France"},
        ]
    )
    s2_test = pd.DataFrame(
        [
            {"entity_id": "S2-T001", "business_name": "Global Tech", "business_address": "500 Tech Boulevard", "country": "US"},
            {"entity_id": "S2-T002", "business_name": "Cafe Paris", "business_address": "12 Rue Rivoli", "country": "France"},
        ]
    )
    s3_test = pd.DataFrame(
        [
            {"entity_id": "S3-T001", "business_name": "Global Technology Inc", "business_address": "500 Tech Blvd", "country": "US"},
        ]
    )

    s1_test.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    s2_test.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    s3_test.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    return tmp_path / "dataset"


# ---------------------------------------------------------
# 1. Data Loader Tests
# ---------------------------------------------------------
def test_loader_all_7_tsvs(mock_dataset_dir: Path):
    train_data, test_data = load_all_7_tsvs(mock_dataset_dir)

    assert len(train_data.s1) == 3
    assert len(train_data.s2) == 2
    assert len(train_data.s3) == 2
    assert train_data.ground_truth is not None
    assert train_data.ground_truth["S1-0001"] == {"S2-0001", "S3-0001"}
    assert train_data.ground_truth["S1-0003"] == set()  # singleton

    assert len(test_data.s1) == 2
    assert len(test_data.s2) == 2
    assert len(test_data.s3) == 1
    assert test_data.ground_truth is None
    assert set(test_data.s1_ids) == {"S1-T001", "S1-T002"}
    assert "S2-T001" in test_data.valid_target_ids


def test_id_list_parsing_and_formatting():
    assert parse_id_list("S2-001, S3-002 , S2-001") == ["S2-001", "S3-002", "S2-001"]
    assert parse_id_list("") == []
    assert parse_id_list(None) == []
    assert parse_id_list(np.nan) == []

    formatted = format_id_list(["S2-001", "S3-002", "S2-001", ""])
    assert formatted == "S2-001,S3-002"


# ---------------------------------------------------------
# 2. Normalization Tests
# ---------------------------------------------------------
def test_normalization_name_address_country():
    # Legal suffix normalization
    assert normalize_name("Acme Corp & Co. Ltd.") == "acme corporation and company limited"
    assert extract_name_core("Acme Corp & Co. Ltd.") == "acme and"

    # Address abbreviations
    norm_addr = normalize_address("123 Main St., Apt 4B, MG Rd.")
    assert "street" in norm_addr
    assert "apartment" in norm_addr
    assert "road" in norm_addr

    # Digits extraction
    assert extract_numbers("Plot 45, Sector 12, PIN 560001") == ["45", "12", "560001"]

    # Open-set country: standardizes aliases but keeps unseen countries intact
    assert normalize_country("usa") == "US"
    assert normalize_country("india") == "India"
    assert normalize_country("france") == "France"
    assert normalize_country("Germany") == "Germany"  # unseen country preserved!


def test_normalize_dataframe():
    df = pd.DataFrame([{"entity_id": "S1-01", "business_name": "Zenith Ltd", "business_address": "10 Park Ave", "country": "US"}])
    norm_df = normalize_dataframe(df)
    assert "norm_business_name" in norm_df.columns
    assert "core_business_name" in norm_df.columns
    assert "name_tokens" in norm_df.columns
    assert "address_numbers" in norm_df.columns
    assert norm_df["norm_business_name"].iloc[0] == "zenith limited"
    assert norm_df["core_business_name"].iloc[0] == "zenith"


# ---------------------------------------------------------
# 3. Multi-Pass Blocking Tests
# ---------------------------------------------------------
def test_multipass_blocking(mock_dataset_dir: Path):
    train_data, _ = load_all_7_tsvs(mock_dataset_dir)
    norm_s1 = normalize_dataframe(train_data.s1)
    norm_targets = normalize_dataframe(train_data.target_records)

    blocker = MultiPassBlocker()
    blocker.fit(norm_targets)
    candidate_map = blocker.block_all(norm_s1)

    assert "S1-0001" in candidate_map
    assert "S1-0002" in candidate_map
    assert "S1-0003" in candidate_map

    # Acme should find both Acme candidates (S2-0001 and S3-0001)
    assert "S2-0001" in candidate_map["S1-0001"]
    assert "S3-0001" in candidate_map["S1-0001"]

    # Evaluate blocking quality
    metrics = evaluate_blocking(candidate_map, train_data.ground_truth, len(norm_targets))
    assert metrics["recall"] == 1.0  # captured all true matches
    assert metrics["reduction_ratio"] > 0.0


# ---------------------------------------------------------
# 4. Pairwise Feature Extraction Tests
# ---------------------------------------------------------
def test_feature_extractor():
    s1_rec = {
        "entity_id": "S1-001",
        "business_name": "Delta Softwares Inc",
        "norm_business_name": "delta softwares incorporated",
        "core_business_name": "delta softwares",
        "business_address": "100 Innovation Way, Suite 200",
        "norm_business_address": "100 innovation way suite 200",
        "country": "US",
        "norm_country": "US",
        "name_tokens": ["delta", "softwares", "incorporated"],
        "address_tokens": ["100", "innovation", "way", "suite", "200"],
        "address_numbers": ["100", "200"],
    }
    cand_match = {
        "entity_id": "S2-001",
        "business_name": "Delta Software",
        "norm_business_name": "delta software",
        "core_business_name": "delta software",
        "business_address": "100 Innovation Way, Ste 200",
        "norm_business_address": "100 innovation way suite 200",
        "country": "US",
        "norm_country": "US",
        "name_tokens": ["delta", "software"],
        "address_tokens": ["100", "innovation", "way", "suite", "200"],
        "address_numbers": ["100", "200"],
    }

    extractor = PairwiseFeatureExtractor()
    feats = extractor.extract_pair(s1_rec, cand_match)

    assert feats["name_token_jaccard"] > 0.0
    assert feats["name_char_3gram_jaccard"] > 0.0
    assert feats["addr_number_jaccard"] == 1.0
    assert feats["country_exact_match"] == 1.0
    assert feats["is_source2"] == 1.0
    assert feats["is_source3"] == 0.0


# ---------------------------------------------------------
# 5. S1 Split Tests
# ---------------------------------------------------------
def test_s1_train_val_split():
    s1_df = pd.DataFrame(
        [
            {"entity_id": f"S1-{i:04d}", "country": "US" if i % 2 == 0 else "India"}
            for i in range(20)
        ]
    )
    gt = {f"S1-{i:04d}": {f"S2-{i:04d}"} if i % 3 != 0 else set() for i in range(20)}

    tr_ids, val_ids = s1_train_val_split(s1_df, ground_truth=gt, val_size=0.25, seed=42)

    # Zero leakage: disjoint sets
    assert set(tr_ids).isdisjoint(set(val_ids))
    assert set(tr_ids).union(set(val_ids)) == set(s1_df["entity_id"])
    assert len(val_ids) == 5

    # 5-fold cross-validation
    folds = s1_kfold_split(s1_df, ground_truth=gt, n_splits=5, seed=42)
    assert len(folds) == 5
    all_val = []
    for tr, val in folds:
        assert set(tr).isdisjoint(set(val))
        all_val.extend(val)
    assert set(all_val) == set(s1_df["entity_id"])


# ---------------------------------------------------------
# 6. Macro F0.5 Metric Tests
# ---------------------------------------------------------
def test_macro_f05_worked_example():
    # Official README:
    # S1-00001: pred = [S2-00047, S2-00193, S3-00812], gt = [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 2/2 = 1.0 -> F_0.5 = 5/7 ≈ 0.7142857
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}

    assert f05_score_single(true_set, pred_set) == pytest.approx(5 / 7, abs=1e-5)

    gt_dict = {"S1-0001": true_set, "S1-0002": set()}
    pred_dict = {"S1-0001": pred_set, "S1-0002": set()}

    # (5/7 + 1.0) / 2 = 6/7 ≈ 0.8571428
    assert macro_f05(gt_dict, pred_dict) == pytest.approx(6 / 7, abs=1e-5)

    eval_summary = evaluate_matching(gt_dict, pred_dict)
    assert eval_summary["macro_f05"] == pytest.approx(6 / 7, abs=1e-5)
    assert eval_summary["singleton_accuracy"] == 1.0
    assert eval_summary["non_singleton_f05"] == pytest.approx(5 / 7, abs=1e-5)


# ---------------------------------------------------------
# 7. Singleton Post-Processor Tests
# ---------------------------------------------------------
def test_singleton_post_processing():
    scores_df = pd.DataFrame(
        [
            {"source1_entity_id": "S1-001", "target_entity_id": "S2-001", "score": 0.85},
            {"source1_entity_id": "S1-001", "target_entity_id": "S3-001", "score": 0.40},
            {"source1_entity_id": "S1-002", "target_entity_id": "S2-002", "score": 0.35},  # weak candidate
        ]
    )
    all_s1 = ["S1-001", "S1-002", "S1-003"]  # S1-003 has no candidates

    processor = SingletonPostProcessor(match_threshold=0.5, singleton_threshold=0.5)
    matches = processor.process(scores_df, all_s1)

    assert matches["S1-001"] == {"S2-001"}
    assert matches["S1-002"] == set()  # gated as singleton because max score < 0.5
    assert matches["S1-003"] == set()  # singleton


# ---------------------------------------------------------
# 8 & 9. Generation and Output Format Tests
# ---------------------------------------------------------
def test_tsv_generation(tmp_path: Path):
    candidate_map = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": ["S2-002"],
        "S1-003": [],
    }
    matching_map = {
        "S1-001": ["S2-001"],
        "S1-002": ["S2-002"],
        "S1-003": [],
    }
    all_s1 = ["S1-001", "S1-002", "S1-003"]

    match_p, cand_p = generate_submission_package(matching_map, candidate_map, all_s1, tmp_path / "output")

    assert match_p.is_file()
    assert cand_p.is_file()

    # Read lines directly to ensure tab separation
    match_lines = match_p.read_text(encoding="utf-8").splitlines()
    assert match_lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert match_lines[1] == "S1-001\tS2-001"
    assert match_lines[3] == "S1-003\t"

    cand_lines = cand_p.read_text(encoding="utf-8").splitlines()
    assert cand_lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert cand_lines[1] == "S1-001\tS2-001,S3-001"
    assert cand_lines[3] == "S1-003\t"


# ---------------------------------------------------------
# 10. Submission Validator Integration Tests
# ---------------------------------------------------------
def test_validator_integration(mock_dataset_dir: Path, tmp_path: Path):
    test_dir = mock_dataset_dir / "test"
    all_s1 = ["S1-T001", "S1-T002"]

    # Valid submission
    matching_map = {"S1-T001": ["S2-T001"], "S1-T002": []}
    candidate_map = {"S1-T001": ["S2-T001", "S3-T001"], "S1-T002": ["S2-T002"]}

    out_dir = tmp_path / "valid_out"
    m_p, c_p = generate_submission_package(matching_map, candidate_map, all_s1, out_dir)

    result = validate_submission(m_p, c_p, test_dir=test_dir)
    assert result.is_valid is True
    assert len(result.errors) == 0

    # Invalid: missing S1 entity row
    bad_m = tmp_path / "bad_m.tsv"
    bad_m.write_text("source1_entity_id\tmatched_entity_ids\nS1-T001\tS2-T001\n")
    bad_res = validate_submission(bad_m, test_dir=test_dir)
    assert bad_res.is_valid is False
    assert any("required S1" in e for e in bad_res.errors)

    # Invalid: self match S1-
    bad_self = tmp_path / "bad_self.tsv"
    bad_self.write_text("source1_entity_id\tmatched_entity_ids\nS1-T001\tS1-T001\nS1-T002\t\n")
    bad_self_res = validate_submission(bad_self, test_dir=test_dir)
    assert bad_self_res.is_valid is False
    assert any("self-matches" in e for e in bad_self_res.errors)


# ---------------------------------------------------------
# 11. End-to-End Pipeline Skeleton Tests
# ---------------------------------------------------------
def test_er_pipeline_skeleton(mock_dataset_dir: Path, tmp_path: Path):
    train_data, test_data = load_all_7_tsvs(mock_dataset_dir)
    pipeline = ERPipeline()

    # Step 1: Prepare data
    norm_s1, norm_targets = pipeline.prepare_data(train_data)
    assert len(norm_s1) == 3
    assert len(norm_targets) == 4

    # Step 2: Blocking
    candidate_map = pipeline.run_blocking(norm_s1, norm_targets)
    assert len(candidate_map) == 3

    # Step 3: Feature extraction
    pairs_df = pipeline.extract_features(candidate_map, norm_s1, norm_targets, ground_truth=train_data.ground_truth)
    assert not pairs_df.empty
    assert "label" in pairs_df.columns

    # Step 4: Baseline scoring
    scores_df = pipeline.score_heuristic_baseline(pairs_df)
    assert "score" in scores_df.columns

    # Step 5: Prediction with singleton gating
    preds = pipeline.predict(scores_df, all_s1_ids=train_data.s1_ids)
    assert len(preds) == 3
    assert isinstance(preds["S1-0001"], set)

    # Step 6: Generate output files
    m_p, c_p = pipeline.generate_outputs(preds, candidate_map, train_data.s1_ids, tmp_path / "train_sub")
    assert m_p.is_file()
    assert c_p.is_file()

    # Step 7: Validate outputs
    val_res = pipeline.validate_outputs(m_p, c_p, required_s1_ids=set(train_data.s1_ids))
    assert val_res.is_valid is True

    # Step 8: Metric score
    score_res = evaluate_matching(train_data.ground_truth, preds)
    assert "macro_f05" in score_res
    assert 0.0 <= score_res["macro_f05"] <= 1.0
