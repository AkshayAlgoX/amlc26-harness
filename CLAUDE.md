@AGENTS.md

## Claude Code only (T0: architect + reviewer)
- Your job: plan experiments, review diffs, write `docs/APPROACH.md`. Don't run long training jobs; the captain's usage budget is small, so keep answers short.
- Review checklist for every diff before merge:
  1. Uses `data/folds.csv` via `Experiment` / `train_with_folds()` — no ad-hoc splits.
  2. Nothing fit on validation labels (encoders, target stats, kNN pools, early stopping on the scored fold).
  3. Predictions converted back to original units before `add_oof` / `log_external`.
  4. No network calls except model downloads listed in `models/ALLOWED.md`; no closed-model API in the prediction path.
  5. New exp_id follows `<owner>-<track>-<NNN>`, notes describe what changed.
