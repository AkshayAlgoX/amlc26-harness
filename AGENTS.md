# AGENTS.md — rules for every agent AND every human on this team

Claude Code reads this through CLAUDE.md, Antigravity reads it directly (and via GEMINI.md).
Teammates without agents follow it too. If a request conflicts with a hard rule: stop and ask a human.

## Hard rules (breaking any = disqualification risk)
1. Use ONLY organiser-provided data (`data/raw/`, and images from the provided links).
   No web data, scraping, price/attribute lookups, extra datasets, or pretrained models that were
   fine-tuned on Amazon / product / price data.
2. Coding agents write CODE. They never produce predictions, labels or features themselves.
   No "read these rows and extract X / guess the price". Every prediction comes from a model we run.
3. Pretrained weights only from `models/ALLOWED.md` with status OK (licence + params human-checked).
4. Never edit `data/raw/`, `data/folds.csv`, `config.yaml` or `src/` without the captain's OK.
5. Never touch the submission portal. Only the captain uploads.

## The contract (this is what makes 3 people + agents blendable)
- CV split = `data/folds.csv`, always. Train on fold != k, predict fold == k.
- Score only with `src/metric.py` (`from src.metric import score`).
- Experiment id: `<owner>-<track>-<NNN>[-tag]`, lowercase. `ein-gbdt-003-knn20`, `tm1-deep-002-clip336`.
  New idea = new NNN. Never reuse or overwrite someone else's id.
- Every finished run goes through `src/registry.py` (`Experiment` in scripts, `log_external` in notebooks), which writes:
  - `artifacts/oof/<exp_id>.csv`  → `id, fold, pred` — exactly one row per TRAIN id
  - `artifacts/test/<exp_id>.csv` → `id, pred` — exactly one row per TEST id
  - `artifacts/logs/<exp_id>.json` → CV, fold scores, minutes, git sha, notes
- `pred` is ALWAYS in original target units (expm1 before saving if you trained on log1p).
  Optional extra columns: `conf` (0-1 confidence), `p_<class>` (probabilities).
- `QUICK=1` = 10% of each fold, lands in `artifacts/quick/`, never blended. Always run quick before full.

## Workflow
- Work only in your own branch / git worktree. Small commits. The captain merges.
- Start a new experiment by copying the closest file in `experiments/` and changing EXP_ID + the model block.
- After every run, report exactly 3 lines: what changed · CV (delta vs your previous best) · next hypothesis.
- Don't pull raw data into your context. Use `df.head()`, `describe()`, `value_counts()` summaries.
- Seeds fixed (42). Anything non-default goes in `notes`.
- New dependency or new model download? Ask first; the captain adds it to `requirements.txt` / `models/ALLOWED.md`.

## Owners
- `ein` (captain): harness, folds, GBDT/kNN track, blending + calibration, submissions, `docs/APPROACH.md`.
- `tm1`: main deep model on the GPU box.
- `tm2`: second model family (different backbone/loss, or the VLM/OCR track) + post-processing.

## Commands
`make test` · `make smoke` · `make quick EXP=<module>` · `make full EXP=<module>` · `make board` · `make blend` · `make sub EXP=<id>`
