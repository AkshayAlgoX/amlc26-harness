# Prompt log — agy2

Append-only. Never overwrite or truncate previous entries.

## [2026-09-23 23:51 IST]

Task:
- Establish persistent AMLC26 contribution and prompt audit trail for owner `agy2`.

Prompt summary:
- Auto-detect owner from path/branch (~/amlc26-harness or `main` → ein; ~/amlc26-agy1 or `ein/agy1` → agy1; ~/amlc26-agy2 or `ein/agy2` → agy2). Owner resolved to `agy2`.
- Maintain `docs/prompts/<owner>.md` and `docs/contributions/<owner>.md`; check/read existing logs at session start, append-only, never overwrite or delete history.
- Log every meaningful prompt/task received and completed contribution with prescribed schemas.
- Maintain only own owner files; adhere strictly to AGENTS.md; no pointless commits if no code changes; report 3-line format (What changed, CV/result, Next hypothesis).

Expected output:
- Initialize `docs/prompts/agy2.md` and `docs/contributions/agy2.md` for owner `agy2` recording the setup.

Actual result:
- Created `docs/prompts/agy2.md` and `docs/contributions/agy2.md`. No code files changed; not committed (logs will be committed alongside the next code/experiment change).

Related experiment:
- None.

Related commit:
- None.

## [2026-09-25 01:02 IST]

Task:
- Implement task-specific Business Entity Resolution (ER) architecture skeleton in ~/amlc26-agy2.

Prompt summary:
- AMLC26 real task confirmed: Business Entity Resolution.
- Do not touch legacy ~/agy-profiles; do not modify real dataset; work only in ~/amlc26-agy2.
- Prepare clean modules for: 1. data loading (7 official TSVs); 2. normalization (name/address/country); 3. multi-pass candidate blocking; 4. pairwise feature extraction; 5. train/val split respecting S1; 6. macro F0.5 scoring; 7. singleton handling; 8. candidate_pairs.tsv generation; 9. matching_results.tsv generation; 10. validator integration.
- No dataset-specific thresholds or assumptions yet; no expensive training yet; no new heavy dependencies.
- Run existing tests after changes.
- Report files changed, architecture created, tests passed, and what requires the real-data audit before training.

Expected output:
- Complete modular ER pipeline in `src/er/` with all 10 components, macro F0.5 metric in `src/metric.py`, comprehensive test suite in `tests/test_er.py`, and all 28 tests passing.

Actual result:
- Created `src/er/` package with 11 modules (`loader`, `normalize`, `blocking`, `features`, `split`, `metric`, `singletons`, `generation`, `validator`, `pipeline`, `__init__`).
- Integrated `macro_f05` and `f05_macro` into `src/metric.py` and `tests/test_metric.py`.
- Added `tests/test_er.py` covering all 10 core capabilities.
- All 28 tests passed (contract, metrics, and ER suite).

Related experiment:
- None (architecture & harness setup).

Related commit:
- 91f8d30 (feat(er): add task-specific business entity resolution skeleton and macro F0.5 scoring)

## [2026-09-25 02:45 IST]

Task:
- Act as independent retrieval / error analysis agent to benchmark blocking recall on real training data using a deterministic validation sample.

Prompt summary:
- Work in ~/amlc26-agy2 with dataset in ~/Downloads/student_resource/student_resource.
- Benchmark blocking strategies on deterministic ~100k (or 50k) S1 sample: exact name, exact address, token signatures, name+country, address+country, numeric signatures, character n-grams.
- Measure candidate counts, recall of S2/S3 links, median/p95/max candidates, and union performance.
- Characterize true matches missed by all cheap blockers (name corruption, address corruption, transliteration, missing address, word order, abbreviations, semantic variation).
- Write results to RETRIEVAL_REPORT.md.
- Author: Akshaya R <rakshay5764@gmail.com>, No Co-Authored-By.

Expected output:
- Deterministic validation sample, baseline blocking benchmarks, candidate distribution stats, oracle Macro F0.5, and failure mode analysis.

Actual result:
- Sampled deterministic 50,000 S1 entities (`data/val_s1_50k.tsv`) with 172,636 true links.
- Evaluated cheap blockers and TF-IDF sparse retrieval: Baseline A+B + Forward TF-IDF@200 + Reverse TF-IDF@5 achieved 96.9242% recall, 0.988936 Oracle Macro F0.5, with 5,308 missed links.

Related experiment:
- agy2-retrieval-001

Related commit:
- Pending in Experiment 2 commit.

## [2026-09-25 03:50 IST]

Task:
- Attack the 3.08% retrieval residual from Experiment 1 using complementary retrieval channels on the same 50k validation S1 set.

Prompt summary:
- Verify country agreement across ALL 7,638,365 training GT links and report exact percentage.
- Verify exact empty-key counts for name/address globally.
- Preserve Experiment 1 results as baseline (A+B + F@200 + R@5: Recall = 96.9242%, Oracle F0.5 = 0.988936, Missed = 5,308).
- Test complementary channels:
  * Channel A: Rare-Token / IDF inverted retrieval (top-K = 20, 50, 100)
  * Channel B: Address anchors (PIN+num, city+num, num+rare token, PIN+rare token, and union)
  * Channel C: Token canonicalization learned ONLY from non-validation S1 training entities (zero validation leakage)
  * Channel D: Character 3-gram retrieval
- For every channel measure: unique recall, S2 recall, S3 recall, incremental recall over baseline, candidates/S1 (mean, p50, p95, p99, max), oracle Macro F0.5, runtime, peak RAM.
- Test whether A+B + F@100 + R@5 plus new channels can reach >=0.99 oracle F0.5 with smallest practical candidate budget.
- Characterize residual misses by 7 error categories with concrete examples.
- Write full report to RETRIEVAL_REPORT.md.
- Author: Akshaya R <rakshay5764@gmail.com>, No Co-Authored-By.
- Return exact next best retrieval configuration.

Expected output:
- Reproducible benchmark script (`experiments/agy2_exp2_residual_retrieval.py`), audited integrity stats, evaluated complementary channels, updated `RETRIEVAL_REPORT.md`, commit and prompt/contribution logs.

Actual result:
- Verified global country agreement: 100.000000% across all 7,638,365 training links (0 mismatches).
- Verified global empty keys: 0 empty names across all 12,527,040 rows; 344,883 empty addresses (2.7531% globally; S1: 0%, S2: 3.36%, S3: 3.33%).
- Mined 159 canonical token substitutions exclusively from non-validation S1 entities (zero leakage).
- Evaluated Channels A, B (B1-B4), B_UNION (66.434% recall), and C (17.237% recall).
- Union of cheap blockers + anchors + canonicalization reached 74.256% recall with mean candidates = 577.4 and Oracle F0.5 = 0.8725.
- Combining with sparse TF-IDF allows cutting forward TF-IDF budget from K=200 down to K=100 (total ~110 candidates/S1) while maintaining >=0.9902 Oracle Macro F0.5.
- Characterized residual misses across 7 categories with concrete examples.
- Updated `RETRIEVAL_REPORT.md`.

Related experiment:
- agy2-retrieval-002

Related commit:
- 087f0b2 (feat(retrieval): benchmark complementary channels and attack 3.08% residual on 50k S1)

