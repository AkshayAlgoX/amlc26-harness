# Prompt log — agy2

Append-only. Never overwrite or truncate previous entries.

## [2026-09-23 23:51 IST]

Task:
- Establish persistent AMLC26 contribution and prompt audit trail for owner `agy2`.

Prompt summary:
- Auto-detect owner from path/branch (~/amlc26-harness or `main` → ein; ~/amlc26-agy1 or `ein/agy1` → agy1; ~/amlc26-agy2 or `ein/agy2` → agy2). Owner resolved to `agy2`.
- Maintain `docs/prompts/<owner>.md` and `docs/contributions/<owner>.md`; check/read existing logs at session start, append-only, never overwrite or delete history.
- Log every meaningful prompt/task received and completed contribution with prescribed schemas.
- Maintain only own owner files; adhere strictly to AGENTS.md; no pointless commits if no code changes; report 3-line format (What changed, CV/result, Next hypothesis).

Expected output:
- Initialize `docs/prompts/agy2.md` and `docs/contributions/agy2.md` for owner `agy2` recording the setup.

Actual result:
- Created `docs/prompts/agy2.md` and `docs/contributions/agy2.md`. No code files changed; not committed (logs will be committed alongside the next code/experiment change).

Related experiment:
- None.

Related commit:
- None.

## [2026-09-25 01:02 IST]

Task:
- Implement task-specific Business Entity Resolution (ER) architecture skeleton in ~/amlc26-agy2.

Prompt summary:
- AMLC26 real task confirmed: Business Entity Resolution.
- Do not touch legacy ~/agy-profiles; do not modify real dataset; work only in ~/amlc26-agy2.
- Prepare clean modules for: 1. data loading (7 official TSVs); 2. normalization (name/address/country); 3. multi-pass candidate blocking; 4. pairwise feature extraction; 5. train/val split respecting S1; 6. macro F0.5 scoring; 7. singleton handling; 8. candidate_pairs.tsv generation; 9. matching_results.tsv generation; 10. validator integration.
- No dataset-specific thresholds or assumptions yet; no expensive training yet; no new heavy dependencies.
- Run existing tests after changes.
- Report files changed, architecture created, tests passed, and what requires the real-data audit before training.

Expected output:
- Complete modular ER pipeline in `src/er/` with all 10 components, macro F0.5 metric in `src/metric.py`, comprehensive test suite in `tests/test_er.py`, and all 28 tests passing.

Actual result:
- Created `src/er/` package with 11 modules (`loader`, `normalize`, `blocking`, `features`, `split`, `metric`, `singletons`, `generation`, `validator`, `pipeline`, `__init__`).
- Integrated `macro_f05` and `f05_macro` into `src/metric.py` and `tests/test_metric.py`.
- Added `tests/test_er.py` covering all 10 core capabilities.
- All 28 tests passed (contract, metrics, and ER suite).

Related experiment:
- None (architecture & harness setup).

Related commit:
- Pending commit alongside contribution log.
