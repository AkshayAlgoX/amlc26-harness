* run `python -m src.board --top 15`
* read newest JSON files in `$AMLC_ARTIFACTS/logs/`
* report best single run and best blend
* flag suspicious CV jumps / fold std / test_mode
* kill tracks with no gain in last 2 full runs
* give one next experiment per owner (ein / tm1 / tm2)
* decide whether to submit
* never propose prohibited external data, closed-model predictions, editing folds, or choosing by public LB over CV
