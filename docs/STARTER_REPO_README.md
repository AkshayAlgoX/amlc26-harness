# amlc26-harness

Shared harness for Amazon ML Challenge 2026. It fixes the four things every teammate and every
agent must agree on: the folds, the metric, the experiment log, and the OOF/test prediction format.
Everything else (models, tools, notebooks, agents) is free choice. Rules: **AGENTS.md**.

## Today (all 3 of you)
```bash
git clone <your-repo> && cd <repo>
pip install -r requirements.txt
make test     # 18 unit/contract tests, ~2 s
make smoke    # full pipeline on fake data, ~1 min: folds -> 2 models -> blend -> validated submission
```
Tested with pandas 2.2 / scikit-learn 1.5 (Colab/Kaggle-like) and pandas 3.0 / scikit-learn 1.8.

## Day 1 — captain, first 60 minutes
1. Put the organiser files in `data/raw/`.
2. Edit `config.yaml` only: column names, task type, metric, submission columns.
3. Open `src/metric.py`, make sure the metric matches the PDF, add the PDF's worked example to `tests/test_metric.py`, run `make test`.
4. `make diagnose` → set `folds.scheme` to the recommendation → `make folds` → commit `data/folds.csv` → push.
5. `make lock` (read-only raw data, folds, config — for agents and humans alike).
6. `make quick EXP=ein_base_001_tfidf_ridge` then `make full EXP=...` → `make sub EXP=ein-base-001-tfidfridge` → upload = format check + first LB anchor.
7. Start the image download on the GPU box: `python scripts/download_images.py --split train --workers 64 --size 512` (then `--split test`).

## Who does what
| | owns | tools |
|---|---|---|
| `ein` (captain) | harness, folds, GBDT + kNN track, blend, submissions, approach doc | Claude Code (review/plan) + 2 Antigravity terminals |
| `tm1` | main deep model (end-to-end fine-tune) on the GPU | anything |
| `tm2` | second model family / VLM-OCR track + post-processing | anything |

Teammates without agents: see `experiments/NOTEBOOK_SNIPPET.md` — train however you like, then one
`log_external(...)` call validates, scores and saves your run.

## The contract in 5 lines
- Split with `data/folds.csv`. Score with `src/metric.py`. Name runs `<owner>-<track>-<NNN>[-tag]`.
- Scripts use `Experiment` (see `experiments/ein_base_001_tfidf_ridge.py`); notebooks use `log_external`.
- Outputs: `artifacts/oof/<id>.csv` (id, fold, pred), `artifacts/test/<id>.csv` (id, pred), `artifacts/logs/<id>.json`.
- `pred` is in original target units. ids are strings. One row per train id / test id — enforced.
- `QUICK=1` runs go to `artifacts/quick/` and are never blended.

## Sharing artifacts between 3 people
Every run writes its own files, so syncing never conflicts. Pick one hub:
- **Shared local worktree**: Set `AMLC_ARTIFACTS=~/amlc26-shared/artifacts` and symlink `data/raw` to `~/amlc26-shared/raw`.
- **GPU box as hub**: everyone copies their `oof/`, `test/`, `logs/` files into `/workspace/amlc/artifacts/`.
- **Shared Drive folder**: in Colab mount Drive and `os.environ["AMLC_ARTIFACTS"] = "/content/drive/MyDrive/amlc26-artifacts"`;
  Kaggle users download their 3 files and drop them in; the captain syncs the folder locally and blends there.

## Commands
| | |
|---|---|
| `make board` | local leaderboard (sorted by CV) + `artifacts/experiments.csv` |
| `make blend` | hill-climb blend of the top 8 full runs on OOF, log-space + calibration for % metrics |
| `python -m src.blend --exps a b c --name cap-blend-007` | blend a chosen set |
| `make sub EXP=cap-blend-003` | write `submissions/cap-blend-003.csv` and validate it |
| `make validate FILE=submissions/x.csv` | re-check any file before upload |
| `python -m src.board --md` | markdown table for `docs/APPROACH.md` |

Log every upload in `submissions/LOG.md` (time, file, CV, public LB). Final pick = best **local CV**, not best public LB.

## Wednesday rehearsal
Use the isolated rehearsal setup in `~/amlc26-rehearsal` so rehearsal never touches:
* real data/raw
* real data/folds.csv
* real shared artifacts
* real experiment IDs
