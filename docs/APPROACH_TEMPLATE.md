# <Team name> — Amazon ML Challenge 2026: approach
<!-- 1–2 pages max. It counts toward final ranking and the top-10 finale pick. Copy to docs/APPROACH.md.
     Fill tables from `python -m src.board --md` and the blend log (artifacts/logs/cap-blend-*.json). -->

## 1. Problem & key findings
Task, data size, metric. The 2–3 EDA findings that shaped the solution.

## 2. Validation
Fold scheme and why (paste the `python -m src.folds diagnose` table). CV vs public LB agreement:

| exp_id | local CV | public LB |
|---|---|---|

## 3. Models
| exp_id | model | inputs | CV |
|---|---|---|---|

## 4. Ensemble & post-processing
Blend members + weights, blend space (log/linear), calibration multiplier, post-processing rules.

## 5. What didn't work
| idea | CV delta | why dropped |
|---|---|---|

## 6. Compliance & reproducibility
- Only organiser-provided data and images were used; no external lookups, scraping, or extra datasets.
- Models used (licence, params): see models/ALLOWED.md — all MIT/Apache-2.0 and ≤ 8B.
- Reproduce: `pip install -r requirements.txt` → `python -m src.folds build` → experiment modules listed above → `python -m src.blend --exps ...` → `python -m src.make_submission --exp <final>`.
