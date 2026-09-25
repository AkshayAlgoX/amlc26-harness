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
