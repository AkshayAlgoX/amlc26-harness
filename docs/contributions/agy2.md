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
