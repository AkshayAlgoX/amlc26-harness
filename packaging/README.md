# AMLC26 Business Entity Resolution — reproduction

Pipeline (one code path for validation and submission): `src/experiments/agy2_pipeline.py`
(retrieval → learned pruner → LightGBM pair matcher) and `src/experiments/agy2_pipeline_merge.py`
(frozen expected-F0.5 set decision → global one-to-one conflict resolution → the two output TSVs → validators).
All numbers were produced on a Kaggle CPU kernel (4 vCPU, 32 GB RAM, internet off) with `requirements.txt`.

## Data
The official files only: `train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`,
read with the production loader (`src/src/er/loader.py`: tab separator, all strings, no NA inference, standard
CSV quoting). No external data, APIs or pretrained models are used. Every model is LightGBM (MIT).

## 1. Train + validate (h4 mode) — produces `export/`
```bash
cd src
PIPELINE_MODE=h4 DATA_ROOT=/path/to/dataset OUT_DIR=/tmp/h4 python experiments/agy2_pipeline.py
```
Reads only `experiments/configs/h4_frozen.json`. It refuses to run if any pinned module's sha256 differs.
It trains the pruner and matcher (5 cross-fitted folds, grouped by S1) on the training sample and selects the
expected-F0.5 decision on out-of-fold scores only. It scores the sealed validation holdout once and writes:
- `export/`: fold models, `decision.json` and a sha256 manifest.
- `artifacts/`: the candidate pool before the budget, and the matcher rows.

Re-training needs the training-sample ID list. It is shipped only if the package was built with
`--include-training-sample`. The shipped `export/` already holds the models used for the submission.

## 2. Test inference (test mode), one shard per run
```bash
for C in US India France; do for I in $(seq 0 $((N_C-1))); do
  PIPELINE_MODE=test MODELS_DIR=export SHARD_COUNTRY=$C SHARD_INDEX=$I SHARD_COUNT=$N_C \
  DATA_ROOT=/path/to/dataset OUT_DIR=/tmp/shards python experiments/agy2_pipeline.py
done; done
```
- **Shards:** each S1 belongs to one shard, set by its country and SHA-256(`amlc26-test-shard-v1|id`) mod N.
  Retrieval statistics are fitted on all test S1 of that country.
- **Why sharding is exact:** an S1's candidates depend only on that S1, all targets, the full S1 corpus of its
  country and target-side key counts. Sharding is therefore exact; this is covered by tests.
- **Fold chain:** each S1 is scored by the pruner and matcher models of its own fold, as in validation.
- **Unseen countries:** France does not occur in training. Its `country` feature is NaN (unknown), so its
  performance was never measured.

## 3. Merge → outputs → validation
```bash
python experiments/agy2_pipeline_merge.py --shards /tmp/shards --models export \
  --data-root /path/to/dataset --out output --organiser-validator utils/validate_submission.py
```
The merge refuses to run unless the shards partition `test_source1` exactly. It then applies the frozen rule,
enforces that each S2/S3 record goes to at most one S1, writes `output/candidate_pairs.tsv` and
`output/matching_results.tsv`, and runs both validators.

`candidate_pairs.tsv` is exactly the set the matcher scored. The training ground truth links every S2/S3 record
to at most one S1 (7,638,365 links, none repeated), so the one-to-one rule follows the task's semantics.
