"""Validate a submission file BEFORE uploading. Exit code 0 = PASS, 1 = FAIL.

  python -m src.validate submissions/cap-blend-003.csv

Checks: exact columns/order (from the sample file), one row per test id, no duplicates,
no missing/NaN/inf, positive values if required, regex format for extraction tasks.
Note: the organisers' sample output may cover only a sample of test rows — we take column
names/order from it, and the row set from test.csv.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from .config import col, data_path, load_config, task
from .data import load_test


def expected_columns() -> list[str]:
    sub_cfg = load_config()["submission"]
    sp = data_path("sample_submission")
    if sp.exists():
        return list(pd.read_csv(sp, nrows=5).columns)
    return [sub_cfg["id_col"], sub_cfg["pred_col"]]


def validate(path) -> list[str]:
    sub_cfg = load_config()["submission"]
    id_col, pred_col = sub_cfg["id_col"], sub_cfg["pred_col"]
    errs: list[str] = []
    sub = pd.read_csv(path, dtype={id_col: str}, keep_default_na=False, na_values=[])
    cols = expected_columns()
    if list(sub.columns) != cols:
        errs.append(f"columns {list(sub.columns)} != expected {cols}")
        if id_col not in sub.columns or pred_col not in sub.columns:
            return errs
    test = load_test()
    test_ids = set(test[col("id")].astype(str))
    ids = sub[id_col].astype(str)
    if len(sub) != len(test):
        errs.append(f"rows {len(sub)} != test rows {len(test)}")
    if ids.duplicated().any():
        errs.append(f"{int(ids.duplicated().sum())} duplicated ids")
    missing, extra = test_ids - set(ids), set(ids) - test_ids
    if missing:
        errs.append(f"{len(missing)} test ids missing (e.g. {sorted(missing)[:3]})")
    if extra:
        errs.append(f"{len(extra)} ids not in test (e.g. {sorted(extra)[:3]})")

    t = task()
    if t["type"] == "regression":
        v = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(float)
        bad = ~np.isfinite(v)
        if bad.any():
            errs.append(f"{int(bad.sum())} empty/non-numeric/inf predictions")
        if t.get("positive_only") and (v[~bad] <= 0).any():
            errs.append(f"{int((v[~bad] <= 0).sum())} predictions <= 0 (must be positive)")
    else:
        vals = sub[pred_col].astype(str)
        if t["type"] == "classification" and (vals.str.strip() == "").any():
            errs.append(f"{int((vals.str.strip() == '').sum())} empty labels")
        rx = t.get("pred_regex")
        if rx:
            pat = re.compile(rx)
            nonempty = vals[vals.str.strip() != ""]
            bad = nonempty[~nonempty.map(lambda s: bool(pat.fullmatch(s)))]
            if len(bad):
                errs.append(f"{len(bad)} predictions fail pred_regex (e.g. {bad.head(3).tolist()})")
    return errs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit("usage: python -m src.validate <submission.csv>")
    errs = validate(argv[0])
    if errs:
        print("FAIL " + argv[0])
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    print("PASS " + argv[0])


if __name__ == "__main__":
    main()
