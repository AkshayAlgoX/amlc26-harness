#!/usr/bin/env bash
# End-to-end check of the whole harness on fake data (~1-2 min on a laptop CPU).
# Run it after cloning, after any change to src/, and on the GPU box before Day 1.
set -euo pipefail
cd "$(dirname "$0")/.."
export AMLC_CONFIG=config.synthetic.yaml
export AMLC_ARTIFACTS="$PWD/artifacts_smoke"
PY=${PY:-python}
rm -rf artifacts_smoke data/synthetic
$PY scripts/make_synthetic.py
$PY -m src.folds diagnose --sample 1000 --dim 64
$PY -m src.folds build
QUICK=1 $PY -m experiments.ein_base_001_tfidf_ridge
$PY -m experiments.ein_base_001_tfidf_ridge
$PY -m experiments.ein_gbdt_001_lgbm_knn
$PY -m src.board
$PY -m src.blend --top 5
$PY -m src.make_submission --exp cap-blend-001 --out submissions/smoke_blend.csv
echo "SMOKE TEST PASSED"
