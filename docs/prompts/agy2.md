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

## [2026-09-25 IST — adversarial review of EXP9 R3 set rule]

Task:
- Review whether R3 (p >= max(0.5, top1 - 0.2)) is a principled set decision or H1-specific.

Prompt summary:
- Check gap-rule collapse, high-cardinality FPs, singleton handling, source/country/cardinality bias, score comparability, calibration need, falsification tests. No retraining, no H4 parameter selection.

Expected output:
- PROVEN / LIKELY / UNKNOWN + the single H4 failure mode to watch.

Actual result:
- Read-only diagnostics on DEV (H1) cross-fitted OOF scores from exp8 c5_matcher/1fa5bcfff7ee922af1fd (script in the session scratchpad; aggregates only; no H2/H3/H4 labels).
- The EXP9 "old gap" baseline is a top-1-only rule, not EXP7's rule, and empties 81.6% of S1 with >= 2 true links.
- R3 vs expF (EXP7 family): -0.00369 macro F0.5 (95% CI -0.00434..-0.00312), lower in 5/5 folds. R3 vs thr0.6: -0.0002 (CI spans 0).
- Scores are approximately calibrated and comparable across top1 bucket, country and source.

Related experiment:
- exp9 (agy1 scratch), exp8 store

Related commit:
- none (no code change)

## [2026-09-25 12:15 IST — prepare fresh H4 for unbiased validation]

Task:
- Prepare completely fresh H4 holdout and frozen R3/retrieval parameters for unbiased validation of the new R3 decision layer.

Prompt summary:
- Create H4 S1 ID list + SHA-256 hash ONLY.
- H4 must be strictly disjoint from H0/H1/H2/H3; no labels/ground truth used for selecting parameters.
- Build/reuse EXP8 artifact graph (C0 normalize, C1 stats/IDF, C2 retrieval superset, C3 pruner OOF, C4 matcher features, C5 matcher OOF, C6 decisions).
- Use corrected production parser and prefix-only ID handling (S1-, S2-, S3-); remove all numeric thresholds.
- Freeze retrieval settings: K_FWD=300, K_REV=10, CAP=1000, exact shard merge.
- R3 parameters proposed by EXP9: floor=0.5, margin=0.2. Do not tune H4 based on labels.
- Guard against H4 overlap, leakage, old ID bug, inconsistent parser.
- Report exact commands/files and readiness; do not run expensive full scoring until AGY1's R3 implementation is merged.

Expected output:
- `expected/h4_s1.ids`, SHA-256 hash, disjointness proof, R3 implementation in `experiments/agy2_exp8_decide.py`, test suite, prep report.

Actual result:
- Generated `expected/h4_s1.ids` (50,000 S1 IDs, salt `amlc26-h4-holdout-v1`, SHA-256 `237a24bb646ac3dd790c01718bf9f1c5226ecdcffec4cfa65b9d169a37ea3a0a`).
- Verified 0 overlap with H0, H1, H2, and H3 (200k excluded union; total 250k pairwise disjoint S1 IDs across all 5 holdouts).
- Integrated `r3` / `dynamic_margin` rule in `experiments/agy2_exp8_decide.py` (`(p >= floor) & (p >= self.mx - margin)`).
- Created `experiments/agy2_exp8_prep_h4.py` and `tests/test_h4_r3.py`.
- 82/82 tests pass in pytest.
- Readiness verified: H4 is sealed in store exclusions, ready for scoring when AGY1's model artifacts/booster are merged.

Related experiment:
- agy2-er-008 / exp9

Related commit:
- see contributions entry

## [2026-09-25 14:40 IST — H1 false-negative mining on exp7 OOF]

Task:
- Mine true links that the exp7 matcher/decision rejects; find high-precision rescue rules; decide whether retraining is justified. H1 only; no retraining first.

Prompt summary:
- Build an FN taxonomy over name/address similarity, JW, char TF-IDF, postal/street number, retrieval rank/score/channels, source, country, candidate-set size and top-score relations. Measure each rescue rule's precision and F0.5 impact on H1 OOF. List unsafe rules.

Expected output:
- Taxonomy, rescue rules with precision and dF0.5, unsafe rules, retraining verdict.

Actual result:
- Kaggle akshayalgox/amlc26-exp7-h1-dump v2 reproduced the frozen exp7 DEV OOF exactly (0.923869046; 110.2 min, 17.8 GB, tests passed). v1 had failed on an empty bundle, caused by a concurrent commit e39f70a.
- Kaggle akshayalgox/amlc26-h1-fn-mining analysed 3,999,866 H1 pairs (aggregates only).
- The matcher is calibrated on DEV. No rescue rule on existing or derived features has positive dF0.5; every candidate is negative in 5/5 folds.
- Local work ran only as a capped systemd poller (400 MB, no swap, 20% CPU).

Related experiment:
- agy2-er-007-h1dump, agy2-er-007-fnmine

Related commit:
- e39f70a (contains agy2_exp7_h1dump.py; committed by a concurrent session)

## [2026-09-25 16:00 IST — EXP9 retrain + Romanization integration review]

Task:
- Complete the pre-registered EXP9 retrain, then review whether the Romanization retrieval lanes (TEST B/C) can be scored downstream using existing artifacts.

Prompt summary:
- Speed-first retrain from cached H1 (B0/B1/B2/B3). Separately, a read-only review of the Romanization result: reuse of exp7/exp9 artifacts, fastest safe downstream measurement, no new expensive job, no H4 tuning.

Expected output:
- Ranked variant table + H4 candidate; PROVEN / CRITICAL RISK / FASTEST NEXT / ARTIFACTS for Romanization.

Actual result:
- EXP9 B2 0.939414 vs B0 0.923869 (see contributions).
- Romanization run saved only aggregate JSON (no pair lists), so its lanes cannot be scored by exp7/exp9 artifacts without regenerating the lane pairs.

Related experiment:
- agy2-er-009, romanization benchmark (concurrent session, untracked code)

Related commit:
- df2091d, 9a44e20

## [2026-09-25 18:50 IST — full cleanup + test-mode pipeline + second ceiling hunt]

Task:
- Remove every procedural blocker before H4 and build the missing test-mode submission pipeline. Then independently re-rank the score-ceiling opportunities. No heavy compute, no H4, no commits.

Prompt summary:
- H4 freeze: B2 + expected-F0.5 only, canonical path, pinned hashes, parity test, fail-closed.
- Deterministic country encoding with unseen France; hygiene; dependencies and packaging.
- Test mode: 1.73M S1, sharded, global conflict resolution, validators; the C0-C6 artifact graph; exact speed-ups only.
- Phase 2: training scale, pruner, B2 FN, singleton/single-match, India, France, Romanization, retrieval, architecture, graph, test scaling.

Expected output:
- Fixes, files, tests, blockers, H4 status, ranked ceiling opportunities, one pre-H4 experiment with exact files and commands.

Actual result:
- Live checks: all Kaggle kernels COMPLETE; none started after 14:44 IST; the "corrected Romanization Kaggle job" never ran. France = 15.0% of test S1 (259,452) and absent from training. Training GT is one-to-one (7,638,365 links, 0 repeated targets).
- Built experiments/agy2_pipeline.py (h4 + test modes; derived by anchored edits from the validated exp9 H4 script), agy2_pipeline_merge.py, agy2_pipeline_freeze.py, configs/h4_frozen.json, scripts/build_submission_package.py, packaging/, docs/PIPELINE_RUNBOOK.md.
- Built exp10 (agy2_exp10_cache_checks.py + configs/exp10_prereg.json; Kaggle kernel files prepared, not pushed).
- Tests: 99 passed locally under a 3 GB cap (the new pipeline tests use synthetic data only).

Related experiment:
- agy2-er-010 (pipeline), agy2-er-010-cache-checks (prepared)

Related commit:
- none (the commit set is proposed to the operator, not committed)
