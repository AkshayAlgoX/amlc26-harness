# Contribution Audit Trail — agy2

## [2026-09-23 23:51:00+05:30]

Task:
- Set up persistent contribution and prompt audit trail for owner `agy2`.

Files changed:
- docs/prompts/agy2.md: Initialized prompt audit trail for owner agy2.
- docs/contributions/agy2.md: Initialized contribution audit trail for owner agy2.

Experiment:
- N/A

Result:
- Initialized directories and audit trail files. Verified branch `ein/agy2` in `/home/akshaya/amlc26-agy2`.

Commit:
- N/A (no code changes; avoiding pointless commit per rules).

Decision:
- Keep.

Next:
- Await next instruction or experiment task.

## [2026-09-25 01:02:00+05:30]

Task:
- Build task-specific Business Entity Resolution (ER) implementation skeleton in ~/amlc26-agy2.

Files changed:
- src/er/__init__.py: Package exports for entity resolution modules.
- src/er/loader.py: TSV data loader and dataclasses for all 7 official files (train_source1/2/3, train_ground_truth, test_source1/2/3).
- src/er/normalize.py: Normalization for business name (legal suffixes, &), address (street abbreviations, numbers), and country (open-set).
- src/er/blocking.py: Composable multi-pass candidate blocking (exact name, prefix, token inverted index, TF-IDF cosine) and blocking evaluation metrics.
- src/er/features.py: Pairwise feature extractor computing 23 similarity features (Jaccard, containment, character n-grams, difflib, numeric/address overlap, country match).
- src/er/split.py: S1-grouped train/validation and K-fold splitting with zero entity leakage and stratification on country/singleton status.
- src/er/metric.py: Macro F0.5 scoring per Source 1 entity reproducing the official worked example and handling singletons.
- src/er/singletons.py: Singleton confidence gating and threshold tuning to guard against precision-damaging false merges.
- src/er/generation.py: Strict TSV generation for candidate_pairs.tsv and matching_results.tsv with format and subset checks.
- src/er/validator.py: Programmatic submission validator mirroring official utils/validate_submission.py rules.
- src/er/pipeline.py: End-to-end ERPipeline wiring loader, normalizer, blocker, feature extractor, scorer, and validator.
- src/metric.py: Registered macro_f05 and f05_macro in METRICS.
- tests/test_metric.py: Added official 2026 worked example test (F0.5 = 5/7) and singleton edge case tests.
- tests/test_er.py: Comprehensive test suite for all 10 ER modules and end-to-end pipeline (12 test functions).

Experiment:
- N/A (framework setup)

Result:
- 28/28 tests passed in pytest (12 contract + 4 metric + 12 ER tests).
- Official worked example verified: F0.5 = 5/7 ≈ 0.7142857.
- Validator integration verified: catches CSV headers, missing S1s, self-matches.

Commit:
- 91f8d30: feat(er): add task-specific business entity resolution skeleton and macro F0.5 scoring

Decision:
- Keep. Architecture ready to consume real data post-audit.

Next:
- Ingest AGY1 data audit findings (name variations, address abbreviations, country distributions, singleton ratio) to tune blocking indices and feature thresholds.

## [2026-09-25 02:45:00+05:30]

Task:
- Benchmark multi-pass blocking retrieval and conduct error analysis on 50k validation S1.

Files changed:
- data/val_s1_50k.tsv: Deterministic sample of 50,000 Source 1 validation entities (seed 42).
- data/val_gt_50k.tsv: Ground truth links for the 50,000 validation entities (172,636 links; 83,407 S2, 89,229 S3).

Experiment:
- agy2-retrieval-001

Result:
- Baseline A+B + Forward TF-IDF@200 + Reverse TF-IDF@5:
  * Overall Recall: 96.9242% (167,328 / 172,636 true links).
  * Oracle Macro F0.5: 0.988936.
  * Missed true links: 5,308 (3.08%).

Commit:
- Bundled into Experiment 2 commit.

Decision:
- Keep as baseline reference.

Next:
- Attack the 3.08% residual using complementary channels (address anchors, learned canonicalization, rare-token indices).

## [2026-09-25 04:04:00+05:30]

Task:
- AMLC26 Experiment 2: Benchmark complementary retrieval channels and attack 3.08% residual on 50k validation S1.

Files changed:
- experiments/agy2_exp2_residual_retrieval.py: Benchmark script evaluating Channels A, B (B1-B4), B_UNION, C (learned token canonicalization), error classifier, and report generator.
- RETRIEVAL_REPORT.md: Comprehensive evaluation report detailing global dataset audit, channel performance, candidate budgets, residual miss characterization, and recommended retrieval pipeline.
- docs/prompts/agy2.md: Updated prompt log for Experiments 1 & 2.
- docs/contributions/agy2.md: Updated contribution log for Experiments 1 & 2.

Experiment:
- agy2-retrieval-002

Result:
- Global Country Agreement: 100.000000% across all 7,638,365 training links (0 cross-country links; country partitioning is 100% loss-free).
- Global Empty Keys: 0 empty names across all 12,527,040 rows; 344,883 empty addresses (2.7531% globally; S1: 0%, S2: 3.36%, S3: 3.33%).
- Channel B_UNION (Address Anchors): 66.434% recall (114,689 / 172,636) with 564 mean candidates/S1.
- Channel C (Learned Token Canonicalization from non-val S1): 17.237% recall (29,758 / 172,636) with 11.2 mean candidates/S1.
- Cheap Blockers Union (A + B + Anchors + Canon): 74.256% recall (128,193 / 172,636) and Oracle Macro F0.5 = 0.872520.
- Optimized Hybrid Configuration (A + B + F@100 + R@5 + Anchors + Canon):
  * Reaches >= 0.9902 Oracle Macro F0.5.
  * Cuts forward TF-IDF candidate budget from K=200 to K=100 (~45% candidate reduction, down to ~110 candidates/S1).
- Peak RAM: 2.22 GB. Execution time: 272.9s streaming 10.3M target rows.
- 28/28 tests pass in pytest.

Commit:
- 087f0b2: feat(retrieval): benchmark complementary channels and attack 3.08% residual on 50k S1

Decision:
- Keep as foundational retrieval pipeline for candidate_pairs generation and ranking.

Next:
- Feed recommended ~110 candidate/S1 blocking pipeline into pairwise feature extractor and train LightGBM ranker.

