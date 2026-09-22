"""Turn an experiment's (or blend's) test predictions into an upload-ready CSV, then validate it.

  python -m src.make_submission --exp cap-blend-003
  python -m src.make_submission --exp ein-gbdt-004 --out submissions/day1_gbdt.csv

Only the captain uploads. Record every upload (time, file, public LB) in submissions/LOG.md.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import ROOT, col, load_config, task
from .data import load_test
from .registry import load_test_preds
from .validate import expected_columns, validate

MIN_POSITIVE = 0.01


def build(exp_id: str, out: str | None = None):
    sub_cfg = load_config()["submission"]
    id_col, pred_col = sub_cfg["id_col"], sub_cfg["pred_col"]
    test = load_test()
    preds = load_test_preds(exp_id)
    df = pd.DataFrame({id_col: test[col("id")].astype(str)})
    df = df.merge(preds[["id", "pred"]].rename(columns={"id": id_col, "pred": pred_col}), on=id_col, how="left")
    if df[pred_col].isna().any():
        sys.exit(f"{int(df[pred_col].isna().sum())} test ids have no prediction in {exp_id}")
    t = task()
    if t["type"] == "regression" and t.get("positive_only"):
        v = df[pred_col].to_numpy(float)
        n_bad = int((v < MIN_POSITIVE).sum())
        if n_bad:
            print(f"WARNING: clipping {n_bad} predictions up to {MIN_POSITIVE}")
        df[pred_col] = np.clip(v, MIN_POSITIVE, None)
    df = df[expected_columns()]
    path = ROOT / (out or f"submissions/{exp_id}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = build(args.exp, args.out)
    errs = validate(path)
    if errs:
        print("FAIL " + str(path))
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    print(f"PASS {path}  -> upload it, then log it in submissions/LOG.md")


if __name__ == "__main__":
    main()
