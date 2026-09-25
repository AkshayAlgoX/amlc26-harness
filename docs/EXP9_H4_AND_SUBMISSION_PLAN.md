# EXP9 — artifact graph, H4 evaluation path and submission path (prepared; nothing here has been run on H4)

## Artifact graph (exp8 content-addressed store; key = sha256(kind, parents, params, code))
| Node | Contents | Built by | Reuse |
|---|---|---|---|
| v0_seed | exp7 H1 dump: queries (50k H1 S1, fold, es, truth_n, country), 3,999,866 cached matcher pairs (43 exp7 features, prune_p/rank, exp7 OOF p), truth table | exp9 retrain (import only) | immutable cache; no retrieval, TF-IDF, pool or pruner recomputation |
| c0_norm | normalised H1 S1 and candidate target texts, street number and PIN, S2+S3 name-token DF | exp9 retrain | every future text-derived feature |
| c4_mfeat | 18 new pair features (groups A, B, C, E), row-aligned to v0 pairs | exp9 retrain | every model variant |
| c5_matcher | per variant: cross-fitted OOF p + 5 fold models | exp9 retrain | decision-rule experiments without retraining |

Location: output of Kaggle kernel `akshayalgox/amlc26-exp9-matcher-retrain` under `/kaggle/working/exp9_store/`. A future kernel attaches it via `kernel_sources`, copies it to a writable store root and opens nodes with `agy2_exp8_store.Store.open(kind, key)`; every open re-verifies file sha256 and parent keys. H3 and H4 IDs are sealed: the store refuses any node containing them.

Not cached: C1 (retrieval statistics) and C2 (full candidate pool). exp7 did not persist them. Any change upstream of the matcher (retrieval, budget, pruner) still needs one full run. That run should persist C1/C2 into the store.

## H4 evaluation (run ONLY after the operator approves one candidate)
1. `python experiments/agy2_exp9_h4_freeze.py experiments/configs/exp9_h4_frozen.json --candidate <B1|B2|B3> --exp9-results <exp9_results.json>`
   - pins the code sha of `agy2_exp9_h4_eval.py`, the SHA-256 of H1/H4/H0-H3, the candidate features and round cap
   - `stats_exclude` = H2, H3, H4 (text statistics never see H4 or earlier holdouts)
2. Commit the frozen config before any H4 scoring.
3. Kaggle kernel (CPU, internet off): `experiments/agy2_exp9_h4_eval.py` with `EXP9_H4_CONFIG`. It:
   - re-derives H4 (salt `amlc26-h4-holdout-v1`, excluding H0-H3) and asserts it equals the frozen list
   - retrains the pruner and the matcher on H1 only (exp7 folds; the candidate's features and cap)
   - re-selects the decision rule on H1 OOF by the exp7 procedure, then scores H4 once
   - estimated ~110-120 min, peak ~18 GB (exp7 reference 17.8 GB at 150k queries; this run has 100k queries)
4. Reproducibility checks reported in the results: H1 OOF inside the run vs the exp9 H1 OOF for the same candidate. The difference comes only from also excluding H4 from text statistics, so it should be small. Also report whether the selected rule and params match exp9.
5. Report exactly as the Phase F H3 report did: retrieval recall, pruned oracle, macro F0.5, P/R, singleton, S2/S3, US/India, cardinality, FN by stage, candidate stats, runtime, peak RAM.

Open decisions for the operator:
- Statistics exclusion. The default excludes H4 (clean). The alternative keeps the exp7 statistics exactly (H2+H3 only), which gives bit-identical H1 features to the cache; H4 text would then enter the IDF (no labels).
- R3. The concurrent session's `agy2_exp8_prep_h4.py` froze the R3 decision rule for H4. On H1 OOF, R3 was worse than the exp7 expected-F0.5 rule (-0.0037, 5/5 folds). This plan uses the exp7 decision procedure.

## Submission path (not built yet; needs operator decisions)
1. Test-time retrieval: the exp7 pipeline on test S1 vs test S2/S3 (production loader). Statistics fitted on test S1 text (transductive, label-free) or on train S1: to decide.
2. Models: pruner and matcher trained on train S1. Options: H1 only (validated as is) or a larger training set (needs a new validation, because more data changes the model).
3. Decision: the frozen rule and params from the approved candidate.
4. Files: `src/er/generation.py` (`generate_candidate_pairs_tsv`: candidate_pairs.tsv = the final-stage matcher candidates, Top-80; `generate_matching_results_tsv`) and `src/er/validator.py` (`validate_submission`).
5. Only the captain uploads (AGENTS.md hard rule 5).
