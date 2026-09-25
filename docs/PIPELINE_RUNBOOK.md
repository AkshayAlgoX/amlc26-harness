# AMLC26 pipeline runbook (H4 validation → test shards → merge → package)

One code path: `experiments/agy2_pipeline.py` (modes `h4`, `test`) + `experiments/agy2_pipeline_merge.py`.
Canonical, fail-closed configuration: `experiments/configs/h4_frozen.json` (B2 features, expected-F0.5 decision).
The previous H4 paths are superseded:
- `expected/agy2_exp9_h4_frozen.json`: untracked; now ignored by git.
- The R3 freeze in `agy2_exp8_prep_h4.py`: retired.

`experiments/agy2_exp9_h4_eval.py` is kept unchanged as the parity reference. `tests/test_pipeline.py` proves that
`agy2_pipeline.py --mode h4` reproduces it exactly, and that its B2 features equal the exp9 cache row by row.

## Artifact graph (what the next full run saves)

| Node | Content | Written by | Valid for reuse when |
|---|---|---|---|
| C0 | normalised texts / numbers / PIN (inside the run) | h4 run (in memory) | — (cheap to rebuild, ~5 min) |
| C1 | per-country TF-IDF + reverse DF mask + address DF | fitted in-run on train S1 minus H2/H3/H4 (h4) or on all test S1 of the shard's country (test) | same `stats_exclude`, same query-free statistics |
| C2+C3 | `artifacts/c2c3_pool_prebudget.parquet`: every retrieved (S1, target) pair **before** the budget with channel bits, forward/reverse score+rank, cheap features, `prune_p`, `prune_rank`, `fold`; H1 labels, H4 labels = -1 | h4 run | any pruner/matcher/budget/lane experiment on H1 (new lanes are unioned as extra rows) |
| C4+C5 | `artifacts/c4c5_matcher_rows.parquet`: budget rows with all 61 matcher features + cross-fitted `p` | h4 run | any decision-layer or matcher-feature experiment |
| C5 models | `export/{pruner,matcher}_fold{0..4}.txt` (best-iteration LightGBM) | h4 run | test inference (fold chain) |
| C6 | `export/decision.json` (rule, params, constrained, budget, feature lists) | h4 run (H1 OOF only) | test merge |
| manifests | `export/manifest.json`, `artifacts/manifest.json`: sha256 + bytes of every file, code sha256 map, config sha256 | h4 run | test mode refuses a changed export or a different code/config |

The exp9 store (v0/c0/c4/c5 in the output of `akshayalgox/amlc26-exp9-matcher-retrain`) stays valid for
matcher-only experiments on the exp7 H1 Top-80 rows. Pools for S1 outside H1+H4 are never cached. A
more-training-data run must add those S1 as extra `dev` queries.

## Run 1 — H4 + export (Kaggle CPU, one kernel)
- inputs: official data, code bundle at the commit that contains the canonical `h4_frozen.json`.
- queries: H1 (50k, train) + H4 (50k, sealed). Statistics exclude H2, H3, H4.
- estimate: exp7 (150k queries) took 108 min and peaked at 17.2 GB. This run has 100k queries plus artifact
  writes, so about 110–120 min and under 18 GB (inferred).
- outputs: `pipeline_h4_results.json` (the only place H4 labels are used), `export/`, `artifacts/`.

## Run 2 — test shards (Kaggle CPU; one kernel per country, shards sequential inside it)
The test set has 1,732,544 S1: US 663,106, India 809,986, France 259,452. S2+S3 have 9,969,589 records.

The shard size is kept at or below the 150k queries exp7 ran at, so the pool per shard stays at or below
exp7's 35.5M pairs and peak RAM at or below exp7's 17.2 GB. Test mode trains nothing, so the peak is lower
(inferred).

| country | S1 | shards | ≈S1/shard |
|---|---|---|---|
| India | 809,986 | 6 | 135k |
| US | 663,106 | 5 | 133k |
| France | 259,452 | 2 | 130k |

- **Exactness:** a shard's candidates depend only on its S1, all targets, the full S1 corpus of the country
  (reverse channel) and target-side key counts. Sharding is therefore exact
  (`tests/test_pipeline.py::test_test_mode_sharding_exact_and_merge_valid`).
- **Speed:** the forward top-K merge skips rows whose maximum new score cannot beat the current threshold. This
  is an exact pre-filter; the parity tests pass with it enabled.
- **Waste accepted for exactness:** each shard re-streams all targets (reverse channel and name DF).
  Per-shard time is about 45–70 min (inferred from exp7's 62 min of retrieval and features at 150k queries),
  so 13 shards take about 10–15 kernel-hours.
- **Session limits:** 3 kernels in parallel fit within Kaggle's 12 h session limit. Whether Kaggle allows 3
  concurrent CPU sessions is unknown; check before launching.
```
PIPELINE_MODE=test MODELS_DIR=<h4 export> SHARD_COUNTRY=India SHARD_INDEX=i SHARD_COUNT=6 OUT_DIR=/kaggle/working/shards
```

## Run 3 — merge (Kaggle CPU)
```
python experiments/agy2_pipeline_merge.py --shards <dir> [--shards <dir> ...] --models <h4 export> \
  --data-root <official data> --out /kaggle/working/output --organiser-validator <utils/validate_submission.py>
```
- The merge works one country at a time: India is about 65M scored rows, roughly 10 GB (inferred).
- Conflict resolution runs globally within each country; the ground truth is one-to-one (proven).
- The TSVs are streamed out in `test_source1` order, and both validators must pass.
- Expected sizes: `candidate_pairs.tsv` up to about 1.8 GB (up to 80 IDs per S1), `matching_results.tsv` far
  smaller. The organiser's zip size limit is unknown.

## Run 4 — package
```
python scripts/build_submission_package.py --team <team> --outputs <merge out> --export <h4 export> \
  --doc <filled Documentation_template.md> --zip <team>_submission.zip [--include-training-sample]
```
The package is built from an allowlist only. `data/`, `expected/`, `exclude_ids/`, `scratch/`, label-derived files
and holdout ID lists can never enter it; this is covered by `tests/test_package.py`.

## Operator decisions still open
1. France (15.0% of test S1s) has no training labels. `country` = NaN, and LightGBM routes it like country=0
   (India) because the training data has no missing countries (proven by test). A country-agnostic B2 variant
   (exp10 arm) would remove the arbitrary routing.
2. Whether the H1 training-sample ID list ships in the package (needed only to re-train).
3. Budget Top-80 (validated) vs Top-40 (half the compute and file size; exp10 arm measures the H1 cost).
4. Whether Kaggle allows three concurrent CPU sessions for Run 2.

## Hardening additions (2026-09-25 evening)
- **B2nc**: `experiments/agy2_pipeline_freeze.py --variant B2nc` sets `drop_features: ["country"]`, which removes
  `country` from the pruner and the matcher. The freeze refuses B2nc and Top-40 unless it is given exp10 results
  with the B2 anchor reproduced and the matching pre-registered decision true.
- **Fail-closed checks:**
  - module sha256;
  - fold contract (`amlc26-exp7-fold-v1`, 5);
  - exact pruner/matcher feature lists;
  - budget ∈ {40, 80};
  - H1/H4 list hashes and H4 re-derivation;
  - the export manifest (test mode requires `mode: h4` and identical feature lists);
  - shard coverage in the merge.
- **dev mode** (`PIPELINE_MODE=dev`): H1 only; H4 is never queried or scored. Its config may be non-canonical,
  built with `dev_config(...)` for retrieval lanes and for extra training S1s drawn disjoint from H0–H4. Lanes are
  refused in h4 and test mode.
- **Merge validation**: `stream_validate` checks, in bounded memory, the headers, coverage and duplicates, that
  every ID has an S2-/S3- prefix and exists in the test set, and that matches ⊆ candidates. It frees merge memory
  first and then runs the vendored organiser validator (`third_party/organiser`) as a separate process.
  - The organiser's ID-existence check is opt-in (`--check-ids`, memory-heavy); the streamed validator always
    does it.
- **Artifacts** also save C0 (normalised S1 and pool-target texts, numbers and PINs) and C1 (per-country
  vectorizers, reverse DF mask, address DF, name-token DF).
  - Target texts are restricted to pool targets: saving truth-only targets would reveal labels.
  - Export manifests carry no H4-derived metric.
- **Kernels**: `scripts/kaggle/make_kernels.py --commit <sha> --bundle-slug <slug> --out <dir>` writes the bundle
  stage (git archive + `BUNDLE_COMMIT`) and five stub kernels. The chain is h4 → test_{US,India,France} → merge.
  All logic is in `scripts/kaggle/kaggle_job.py`, which checks the commit, the config sha and the tests before
  doing anything. Nothing is pushed automatically.
- **Post-exp10 experiment scripts** (Kaggle, cache or saved-artifact only):
  - `agy2_mine_b2_errors.py` (India-first B2 error mining);
  - `agy2_exp11_seed_stability.py` (noise floor);
  - `agy2_pruner_from_pool.py` (pruner variants from the saved pool; its base variant reproduces the stored scores
    exactly);
  - `agy2_lane_romanization.py` (leakage-safe lane node), followed by a dev-mode pipeline run for its downstream
    effect.

## Exact order after exp10
1. Read `exp10_results.json`. If `anchor.b2_reproduced` is false, stop: no decision may be taken.
2. Re-freeze with the pre-registered decisions:
   `agy2_pipeline_freeze.py --exp9-results ... [--variant B2nc] [--top-n 40] --exp10-results ...`.
   Re-run the tests, commit, and build the bundle and kernels from that commit.
3. H4 kernel (`amlc26-p-h4`): the only use of H4 labels. It exports the production models and the C0–C6
   artifacts.
4. Test kernels (US, India, France) run in parallel if Kaggle allows. Then the merge kernel produces
   `output/*.tsv` and the validators.
5. Fill in `docs/Documentation_AMLC26.md`, build the package, and hand over to the captain for upload.
6. Only if time remains: cached ceiling experiments (mining → India features; seed floor; lanes via dev mode).
   Any change adopted after H4 is backed only by H1 out-of-fold evidence.

## H4 candidate after exp10 (frozen 2026-09-25 20:40 IST)
- exp10 (pre-registered, commit 75c82b7):
  - B2 anchor reproduced exactly;
  - country-agnostic rejected (−0.00066, single-match −0.0118);
  - Top-40 rejected (single-match −0.0111);
  - learning curve +0.0041 / +0.0043 per doubling of training S1, so a larger training set is justified.
- Canonical `h4_frozen.json`: **B2, Top-80, expected-F0.5, `country` kept, trained on H1 + 50,000 extra
  training S1** (`extra_dev` = {n: 50000, salt: `amlc26-extra-train-v1`}; salted SHA-256 draw disjoint from
  H0–H4; the draw's sha256 is recorded in the results).
  - The freeze refuses `--extra-train` without the exp10 learning-curve decision, and above 50,000.
  - The H4 run queries 150k S1 (100k training + 50k H4), which is the exp7 envelope (17.2 GB).
  - The expected gain from the larger training set is an extrapolation of the measured slope, not a measurement;
    H4 is its first fresh test.
