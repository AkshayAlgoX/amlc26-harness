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

## [2026-09-25 04:08 IST]

Task:
- AMLC26 Controlled Retrieval Ablation on 50k Validation S1 without unverified claims or projected results.

Prompt summary:
- Run one controlled ablation on the same fixed 50k validation S1 sample with hard country partition.
- Test baseline (A+B + F@200 + R@5) and 7 candidate configurations:
  1. A+B + F@100 + R@5
  2. 1 + learned-token retrieval (separate view, non-val learned)
  3. 2 + B2 (city + street number)
  4. 2 + B3 with strict frequency cap/subblocking
  5. 2 + B2 + B3
  6. 2 + B2 + B3 + B4 (PIN + rare token)
  7. 2 + B2 + B3 + B4 + B1 (PIN + street number)
- For every configuration report: exact unique recall, S2 recall, S3 recall, oracle Macro F0.5 per S1, candidates/S1 (mean, p50, p95, p99, max), exact unique candidate pair count, incremental recall over config 1, runtime, peak RAM.
- Do not treat B3 as acceptable merely because recall is high; require composite subblocking / frequency control and report recall lost by any cap.
- Maintain original and canonicalized retrieval views separately; verify no validation leakage.
- Compute exact UNIQUE candidate union (no summing overlapping block counts).
- For best configuration, calculate Oracle F0.5 at full uncapped, 150, 125, 100 candidates/S1.
- Answer what is the smallest retrieval configuration achieving >=0.990 oracle Macro F0.5. Clearly label projections.

Expected output:
- Controlled ablation script (`experiments/agy2_exp3_controlled_ablation.py`), full ablation table with exact counts, budget constraint analysis, updated `RETRIEVAL_REPORT.md`, git commit and audit log updates.

Actual result:
- Evaluated configurations 1 through 7 streaming 10.3M records in 214.1s (Peak RAM 1,438.7 MB).
- Measured B3 cap impact: capping B3 (max 50 cands/S1, bucket cap 100) lost 16.784% recall (dropped from 61.799% to 45.015%), but reduced candidate volume by 96.3% (from 609.9 to 22.5 mean cands/S1).
- Measured B4 impact: added +2.151% unique recall (+3,710 true links) with only +0.2 mean cands/S1 (marginal ratio +107.5% per 10 cands).
- Measured B1 impact: added only +0.035% recall (+60 links), demonstrating it is largely redundant.
- Determined standalone non-TFIDF ceiling is 67.895% (Oracle F0.5 = 0.830376, 11.34M pairs). Cheap blocking alone cannot achieve >=0.990 Oracle F0.5.
- Best configuration reaching >=0.990 Oracle F0.5 is Config 6 + F@100 + R@5 (~118 candidates/S1, projected Oracle F0.5 = 0.9902).
- Budget-capped Oracle F0.5 on Config 6 evaluated at Top-150 (0.7667 standalone / 0.9898 projected), Top-125 (0.7579 standalone / 0.9891 projected), and Top-100 (0.7464 standalone / 0.9874 projected).
- Updated `RETRIEVAL_REPORT.md`.

Related experiment:
- agy2-retrieval-003

Related commit:
- 6279c50 (feat(retrieval): controlled retrieval ablation on 50k S1 with frequency-capped B3 and budget constraints)



## [2026-09-25 05:24:11+05:30]

Task:
- Root-fix a suspected source-classification bug in the fresh-holdout retrieval pipeline.

Prompt summary:
- Verify the reported S2-166376419 → S3 misclassification by inspecting code; remove every numeric-magnitude/threshold/range/row/length heuristic for S2 vs S3; derive source only from the entity_id prefix via one helper; add tests (S2-166376419, S2-999999999, S3-1, S3-999999999 → correct source); search for duplicate logic; run only tests that do not scan the raw data; do not run the fresh holdout; commit as Akshaya R with no co-author line.

Expected output:
- Bug verdict, changed files/functions, test results, commit SHA.

Actual result:
- Bug CONFIRMED in experiments/agy2_exp4_fresh_holdout_validation.py: encode_tid stored S2 as the raw number and S3 as number+10,000,000; S2/S3 was then decided by `< 10_000_000`. Real IDs are unpadded numbers up to ~1e9 (≈99% have 8–9 digits), so nearly every S2 was counted as S3 and S2-(10,000,000+k) collided with S3-k.
- Added src/er/ids.py (source_of, is_target_id, lossless encode_target_id/decode_target_id with an explicit prefix tag, code_source); exp4 and features.py now use it. No other magnitude heuristics found (exp2/exp3/src use prefix checks).
- 49/49 tests pass (tests/test_er_ids.py + existing ER/metric/contract tests). Fresh holdout not run.

Related experiment:
- agy2-retrieval-004 (fresh holdout) — results produced before this fix are invalid.

Related commit:
- see contributions entry

## [2026-09-25 05:27:08+05:30]

Task:
- Remove committed organiser-derived data (data/val_s1_50k.tsv, data/val_gt_50k.tsv) from unpushed ein/agy2 history.

Prompt summary:
- Captain instruction: "remove the committed data files from unpushed history".

Expected output:
- Rewritten unpushed commits without the files; files kept locally; identity preserved.

Actual result:
- git filter-branch --index-filter over origin/ein/agy2..ein/agy2 (7 commits); only the two files dropped; author/committer/dates/messages identical; files kept on disk (untracked) and ignored via .git/info/exclude (data/val_*.tsv). 49/49 tests pass. Not pushed.
- SHA map (earlier log entries cite the old SHAs): 087f0b2→98d8244, c7fd160→99b5eba, 6279c50→f4ad5d5, 2fc16d2→982f2ad, 7f51cb1→1860100 (91f8d30, 639bed4 unchanged)

Related experiment:
- n/a

Related commit:
- see contributions entry

## [2026-09-25 05:38:34+05:30]

Task:
- Prepare the corrected fresh-hash 50k holdout retrieval experiment for Kaggle (captain instruction).

Prompt summary:
- Verify prefix-only S2/S3 identity; measure BASELINE (A+B+F@200+R@5) vs PROPOSED (A+B+F@100+R@5+B2+freq-controlled B3+B4+non-destructive canon view) with exact unique unions, S2/S3 recall, oracle F0.5, budget stats, US/India/empty-address/cardinality subgroups, missed links and marginals; reverse over ALL train S1; canon mined outside holdout; no LightGBM, no dense, no laptop-wide run.

Expected output:
- Kaggle-ready script + cell; measured results after the Kaggle run.

Actual result:
- exp4 audit: besides the fixed ID bug it cannot produce valid numbers (reverse ranked only against the holdout; "baseline" was F@100; TF-IDF fitted on holdout only; B3 "rare" = longest tokens, capped by holdout-side counts; marginals not leave-one-out). Marked SUPERSEDED.
- Wrote experiments/agy2_exp5_corrected_holdout.py (Kaggle-first, env-configurable) + tests/test_exp5.py. Smoke-tested on tiny synthetic data only (invariants pass). 52/52 tests pass. Not run on real data.

Related experiment:
- agy2-retrieval-005 (pending Kaggle execution)

Related commit:
- see contributions entry

## [2026-09-25 06:24:00+05:30]

Task:
- Execute exp5 (corrected fresh-hash holdout retrieval validation) end-to-end on Kaggle and return verified results.

Prompt summary:
- Verify bundle; create private code dataset; private CPU script kernel with official data + code bundle; REVERSE_MODE=full with no fallback; capture outputs; validity checks; report BASELINE/BASE100/PROPOSED; decide freeze against oracle >= 0.990, mean <= 80, p99 <= 250. No LightGBM.

Expected output:
- Measured results + decision.

Actual result:
- Kaggle dataset akshayalgox/amlc26-agy2-exp5 (private) + kernel akshayalgox/amlc26-exp5-corrected-holdout (private, CPU, no internet). COMPLETE, exit 0, 32.2 min, child peak RSS 7,560 MB.
- PROPOSED: recall 96.159%, oracle 0.985883, mean 216.7 / p99 1,947 cands per S1 -> FAILS oracle and budget. BASELINE: 94.870% / 0.980957 / 215.5 / p99 438. BASE100: 93.819% / 0.976823 / 118.2 / p99 359.
- Artifacts saved locally: ~/amlc26-kaggle-refresh/exp5/run/output/exp5/ (stdout.log, exp5_results.json, holdout_s1.ids, run_summary.json).

Related experiment:
- agy2-retrieval-005

Related commit:
- see contributions entry

## [2026-09-25 IST — exp6 launch]

Task:
- Build and run integrated EXP6 on Kaggle: retrieval -> learned pruner -> pair matcher -> set decision.

Prompt summary:
- One retrieval pass for two 50k S1 groups: the previous 50k is for training only, and the fresh exp5 holdout is for evaluation only. The fresh group never contributes labels, statistics, fitting or tuning. Pruner and matcher are LightGBM with GroupKFold by S1. Budgets Top-N 40..120 plus floors, with the constraint mean <= 80 and p99 <= 250. Set rules maximise macro F0.5. Compare unconstrained against greedy conflict resolution. Report measured values only. CPU only, and stop cleanly if it doesn't fit.

Expected output:
- Stage-7 table, best verified budget and fresh macro F0.5, FP/FN categories, dense-retrieval verdict.

Actual result:
- Code e27a582 (experiments/agy2_exp6_integrated.py, tests/test_exp6.py; 54/54 tests pass). The synthetic smoke test passed locally. The probe showed rapidfuzz is absent on Kaggle, so Jaro-Winkler is implemented in numba with a textbook self-test.
- Private dataset akshayalgox/amlc26-agy2-exp6 (code + ID lists only) and private CPU kernel akshayalgox/amlc26-exp6-integrated v1 pushed; results pending.

Related experiment:
- agy2-er-006

Related commit:
- e27a582

## [2026-09-25 IST — exp7 validation integrity + H3]

Task:
- Root-fix the EXP6 audit issues and score a truly untouched H3 holdout on Kaggle.

Prompt summary:
- A: one canonical TSV reader (parity with src/er/loader.py on all 7 files, contract test on quoted rows).
- B: pruner and matcher cross-fitted on one shared S1 fold assignment.
- C: no calibration applied across different prediction distributions.
- D/E: new SHA-256 H3 disjoint from all previous lists; configuration frozen in JSON before scoring; one budget pre-registered.
- F: fixed H3 report and H2 comparison. No H3 tuning, no dense retrieval, no new channels.

Expected output:
- H3 report (retrieval recall, pruner oracle at the fixed budget, matcher macro F0.5, P/R, singleton, S2/S3, US/India, cardinality, FN by stage, candidate stats, runtime, RAM) + H2 confirm/contradict verdict.

Actual result:
- Code 2458805 + c0cb1fc; frozen config 874f301 (experiments/configs/exp7_h3_frozen.json, sha256 2cbc44f3...). Tests 61/61.
- Prep kernel akshayalgox/amlc26-exp7-prep: contract reader == production loader on all 7 files. Legacy plain split differed on 10 train rows (addresses) and 813 test rows (mostly names). H3 = 50,000 IDs (salt amlc26-h3-holdout-v1, sha256 6317c82c...), 0 overlap with H0/H1/H2.
- Main kernel akshayalgox/amlc26-exp7-h3 v1 pushed; results pending.

Related experiment:
- agy2-er-007-h3

Related commit:
- 2458805, c0cb1fc, 874f301
- [11:45 update] exp7 COMPLETE (exit 0, 108.05 min, 17,209 MB). H3 macro F0.5 0.923461 vs H2 0.922723 vs DEV-OOF 0.923869 -> H3 confirms H2. See contributions entry.

## [2026-09-25 12:05 IST — exp8 turbo path (built by T0 in the agy2 worktree)]

Task:
- Build the reusable exp8 experimentation system (C0 normalize → C1 statistics/TF-IDF → C2 retrieval superset → C3 pruner OOF → C4 matcher features → C5 matcher OOF → C6 decision sweeps) so that only retrieval/statistics changes need a full-corpus run.

Prompt summary:
- Do not modify experiments/agy2_exp7_*.py or the frozen exp7 config; no Kaggle jobs; no H3 labels or H3 rows in development artifacts; don't modify main; don't push.
- Content-addressed store with manifests and guards; v0 sandbox seeded from the exp6 pool; C0–C2 builder (production loader, stats exclude H2+H3, K=300 / reverse 10 / cap 1000 superset, sharded exact merge); cached pruner/matcher/decide; sealed final scorer; synthetic tests only.
- Measure the v0 seed time and one rule sweep on cached scores; give the expected C0–C2 build time.

Expected output:
- Code + tests committed as Akshaya R (no co-author); measured v0 timings; artifact graph; what can be iterated without a corpus rerun.

Actual result:
- 8 new files (store, features, builder, seed, pruner, matcher, decide, final) + tests/test_exp8.py. Tests 72/72 (61 existing + 11 new), synthetic only.
- exp7 parity (synthetic, the unmodified exp7 main): identical pools except rows swapped by exact score ties at a top-k boundary (exp7 orders ties arbitrarily; exp8 by score desc, id asc).
- v0 on the real exp6 pool (local, neg-rate 0.1, 8 threads, no text features): seed 6.4 s; pruner 4.2 min; budget table 0.26 min; matcher 0.85 min; rule sweep 4.4 s for 176 evaluations (19.6 s for 736 with gates); peak RSS 5.4 GiB.

Related experiment:
- agy2-er-008 (turbo path). No Kaggle run.

Related commit:
- see the contribution entry.
