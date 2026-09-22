# data/
- `raw/` — organiser files exactly as downloaded (train.csv, test.csv, sample_test_out.csv, ...). Never edited, never committed. `make lock` makes it read-only.
- `folds.csv` — built ONCE by the captain (`python -m src.folds build`), committed, never edited.
- `images/` — output of `scripts/download_images.py` (not committed).
- `synthetic/` — fake data for `make smoke` only.
