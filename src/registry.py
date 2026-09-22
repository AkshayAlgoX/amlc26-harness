"""The contract. Every experiment — script, notebook, agent or human — ends here.

Two entry points:

1) Experiment  (for scripts that loop over folds.csv themselves)
    exp = Experiment("ein-gbdt-003-knn20", notes="LGBM + 20-NN target feats")
    for fold, trn, val in exp.split():
        ...fit on trn, predict val and exp.test...
        exp.add_oof(fold, val[ID], val_pred)
        exp.add_test_fold(test_pred)          # averaged over folds at finish()
    exp.finish()

2) log_external  (for notebooks / any tool: you already have OOF + test predictions)
    log_external("tm1-deep-001-clip336", oof=oof_df, test=test_df, notes="...")
    oof_df:  columns id, pred  — one row per TRAIN id (fold is attached from folds.csv)
    test_df: columns id, pred  — one row per TEST id

Outputs (under $AMLC_ARTIFACTS or config paths.artifacts):
    oof/<exp_id>.csv    id, fold, pred[, conf, p_*]
    test/<exp_id>.csv   id, pred[, conf, p_*]
    logs/<exp_id>.json  cv, fold scores, minutes, git sha, notes
QUICK=1 runs go to quick/ and are never blended.

Rules: pred is ALWAYS in original target units (never log space). ids are strings.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, artifacts_dir, col, load_config, task
from .data import load_folds, load_test, train_with_folds
from .metric import greater_is_better, score

EXP_ID_RE = re.compile(r"^[a-z0-9]+-[a-z0-9]+-\d{3}(?:-[a-z0-9]+)*$")


def check_exp_id(exp_id: str) -> tuple[str, str]:
    if not EXP_ID_RE.match(exp_id):
        raise ValueError(f"exp_id '{exp_id}' must look like <owner>-<track>-<NNN>[-tag], e.g. ein-gbdt-003-knn20")
    owner, track = exp_id.split("-")[:2]
    return owner, track


def dirs() -> dict[str, Path]:
    base = artifacts_dir()
    d = {k: base / k for k in ("oof", "test", "logs", "quick")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _git() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return sha, dirty
    except Exception:
        return "nogit", False


def _is_numeric_task() -> bool:
    return task()["type"] == "regression"


def read_preds(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False, na_values=[])
    df["id"] = df["id"].astype(str)
    if _is_numeric_task():
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    else:
        df["pred"] = df["pred"].astype(str)
    return df


def _validate_preds(df: pd.DataFrame, expected_ids: pd.Series, where: str) -> list[str]:
    errs = []
    if "id" not in df.columns or "pred" not in df.columns:
        return [f"{where}: needs columns 'id' and 'pred' (got {list(df.columns)})"]
    ids = df["id"].astype(str)
    if ids.duplicated().any():
        errs.append(f"{where}: {int(ids.duplicated().sum())} duplicated ids")
    exp_set, got_set = set(expected_ids.astype(str)), set(ids)
    if exp_set != got_set:
        errs.append(f"{where}: id mismatch — missing {len(exp_set - got_set)}, unexpected {len(got_set - exp_set)}")
    if _is_numeric_task():
        p = pd.to_numeric(df["pred"], errors="coerce").to_numpy(float)
        bad = ~np.isfinite(p)
        if bad.any():
            errs.append(f"{where}: {int(bad.sum())} non-numeric/NaN/inf predictions")
    return errs


def _log_path(exp_id: str, quick: bool) -> Path:
    d = dirs()
    return (d["quick"] / f"log_{exp_id}.json") if quick else (d["logs"] / f"{exp_id}.json")


def _write(exp_id: str, oof: pd.DataFrame, test: pd.DataFrame | None, record: dict, quick: bool) -> None:
    d = dirs()
    if quick:
        oof.to_csv(d["quick"] / f"oof_{exp_id}.csv", index=False)
        if test is not None:
            test.to_csv(d["quick"] / f"test_{exp_id}.csv", index=False)
    else:
        oof.to_csv(d["oof"] / f"{exp_id}.csv", index=False)
        test.to_csv(d["test"] / f"{exp_id}.csv", index=False)
    with open(_log_path(exp_id, quick), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=float)


def _score_oof(oof: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, list[float]]:
    """oof has id, fold, pred; truth has id, y. Returns (overall, per-fold)."""
    m = oof.merge(truth, on="id", how="left", validate="one_to_one")
    overall = score(m["y"].to_numpy(), m["pred"].to_numpy())
    per_fold = [score(g["y"].to_numpy(), g["pred"].to_numpy()) for _, g in m.groupby("fold", sort=True)]
    return overall, per_fold


def _record(exp_id, kind, oof, per_fold, overall, minutes, notes, quick, n_test, extra=None) -> dict:
    owner, track = check_exp_id(exp_id)
    sha, dirty = _git()
    rec = {
        "exp_id": exp_id, "owner": owner, "track": track, "kind": kind, "quick": quick,
        "metric": task()["metric"], "greater_is_better": greater_is_better(),
        "cv": round(float(overall), 6),
        "cv_mean": round(float(np.mean(per_fold)), 6), "cv_std": round(float(np.std(per_fold)), 6),
        "fold_scores": [round(float(s), 6) for s in per_fold],
        "n_oof": int(len(oof)), "n_test": int(n_test), "minutes": round(float(minutes), 2),
        "git_sha": sha, "git_dirty": dirty,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": Path(load_config()["_config_path"]).name, "notes": notes,
    }
    if extra:
        rec.update(extra)
    return rec


def _refuse_overwrite(exp_id: str, quick: bool) -> None:
    if not quick and _log_path(exp_id, False).exists() and os.environ.get("OVERWRITE") != "1":
        raise FileExistsError(f"{exp_id} already logged. New idea = new NNN. (OVERWRITE=1 to replace your own run.)")


def _agg(parts: list[np.ndarray], how: str):
    if _is_numeric_task():
        arr = np.vstack([np.asarray(p, dtype=np.float64) for p in parts])
        if how == "geo":
            return np.expm1(np.log1p(np.clip(arr, 0, None)).mean(axis=0))
        return arr.mean(axis=0)
    # labels / strings: majority vote across folds (ties -> earliest fold)
    arr = np.vstack([np.asarray(p, dtype=object) for p in parts]).T
    return np.array([Counter(row).most_common(1)[0][0] for row in arr], dtype=object)


class Experiment:
    def __init__(self, exp_id: str, notes: str = "", test_agg: str = "mean"):
        check_exp_id(exp_id)
        self.exp_id, self.notes, self.test_agg = exp_id, notes, test_agg
        self.quick = os.environ.get("QUICK") == "1"
        _refuse_overwrite(exp_id, self.quick)
        self.cfg = load_config()
        self.id_col, self.target = col("id"), col("target")
        self.train = train_with_folds()
        self.test = load_test()
        if self.quick:
            frac, seed = float(self.cfg["quick"]["frac"]), int(self.cfg["folds"]["seed"])
            self.train = (self.train.groupby("fold", group_keys=False)
                          .sample(frac=frac, random_state=seed).reset_index(drop=True))
            self.test = self.test.sample(frac=frac, random_state=seed).reset_index(drop=True)
            print(f"[QUICK] {len(self.train)} train rows, {len(self.test)} test rows -> artifacts/quick/")
        self._oof: list[pd.DataFrame] = []
        self._test_parts: list[np.ndarray] = []
        self._test_final = None
        self._t0 = time.time()

    @property
    def n_splits(self) -> int:
        return int(self.train["fold"].max()) + 1

    def split(self):
        """Yields (fold, train_df, valid_df) using data/folds.csv."""
        for k in range(self.n_splits):
            trn = self.train[self.train["fold"] != k].reset_index(drop=True)
            val = self.train[self.train["fold"] == k].reset_index(drop=True)
            yield k, trn, val

    def split_idx(self):
        """Yields (fold, trn_idx, val_idx) as row positions into self.train — handy when you
        precompute embeddings/features for all rows once."""
        f = self.train["fold"].to_numpy()
        for k in range(self.n_splits):
            yield k, np.where(f != k)[0], np.where(f == k)[0]

    def add_oof(self, fold: int, ids, preds, **extra_cols) -> None:
        df = pd.DataFrame({"id": np.asarray(ids).astype(str), "fold": int(fold), "pred": np.asarray(preds)})
        for k, v in extra_cols.items():
            df[k] = np.asarray(v)
        self._oof.append(df)

    def add_test_fold(self, preds) -> None:
        if len(preds) != len(self.test):
            raise ValueError(f"test preds length {len(preds)} != test rows {len(self.test)}")
        self._test_parts.append(np.asarray(preds))

    def set_test(self, preds) -> None:
        """Use instead of add_test_fold when you refit on all data (note it in notes)."""
        if len(preds) != len(self.test):
            raise ValueError(f"test preds length {len(preds)} != test rows {len(self.test)}")
        self._test_final = np.asarray(preds)

    def finish(self) -> dict:
        if not self._oof:
            raise RuntimeError("no OOF predictions added")
        oof = pd.concat(self._oof, ignore_index=True)
        if self._test_final is not None:
            test_pred, mode = self._test_final, "full_refit"
        elif self._test_parts:
            test_pred, mode = _agg(self._test_parts, self.test_agg), f"fold_{self.test_agg}"
        else:
            raise RuntimeError("no test predictions: call add_test_fold() per fold or set_test() once")
        test = pd.DataFrame({"id": self.test[self.id_col].astype(str).to_numpy(), "pred": test_pred})
        errs = _validate_preds(oof, self.train[self.id_col], "OOF") + _validate_preds(test, self.test[self.id_col], "TEST")
        if errs:
            raise ValueError("contract violated:\n  " + "\n  ".join(errs))
        truth = pd.DataFrame({"id": self.train[self.id_col].astype(str), "y": self.train[self.target]})
        overall, per_fold = _score_oof(oof, truth)
        minutes = (time.time() - self._t0) / 60
        rec = _record(self.exp_id, "model", oof, per_fold, overall, minutes, self.notes, self.quick,
                      len(test), {"test_mode": mode})
        _write(self.exp_id, oof, test, rec, self.quick)
        _print(rec)
        return rec


def log_external(exp_id: str, oof: pd.DataFrame, test: pd.DataFrame, notes: str = "",
                 minutes: float | None = None, quick: bool = False) -> dict:
    """Register predictions produced anywhere (Colab, Kaggle, a teammate's notebook).
    You MUST have trained with data/folds.csv; the fold column is attached here from that file."""
    check_exp_id(exp_id)
    _refuse_overwrite(exp_id, quick)
    oof = oof.copy()
    test = test.copy()
    oof["id"], test["id"] = oof["id"].astype(str), test["id"].astype(str)
    tr = train_with_folds()
    te = load_test()
    if quick:
        tr = tr[tr[col("id")].isin(oof["id"])]
        te = te[te[col("id")].isin(test["id"])]
    folds = load_folds()
    oof = oof.drop(columns=["fold"], errors="ignore").merge(folds, on="id", how="left")
    if oof["fold"].isna().any():
        raise ValueError(f"{int(oof['fold'].isna().sum())} OOF ids not in folds.csv")
    oof["fold"] = oof["fold"].astype(int)
    cols = ["id", "fold", "pred"] + [c for c in oof.columns if c not in ("id", "fold", "pred")]
    oof = oof[cols]
    errs = _validate_preds(oof, tr[col("id")], "OOF") + _validate_preds(test, te[col("id")], "TEST")
    if errs:
        raise ValueError("contract violated:\n  " + "\n  ".join(errs))
    truth = pd.DataFrame({"id": tr[col("id")].astype(str), "y": tr[col("target")]})
    overall, per_fold = _score_oof(oof, truth)
    rec = _record(exp_id, "model", oof, per_fold, overall, minutes or 0.0, notes, quick, len(test),
                  {"test_mode": "external"})
    _write(exp_id, oof, test, rec, quick)
    _print(rec)
    return rec


def _print(rec: dict) -> None:
    arrow = "higher=better" if rec["greater_is_better"] else "lower=better"
    tag = "[QUICK] " if rec["quick"] else ""
    print(f"{tag}{rec['exp_id']}  {rec['metric']}={rec['cv']:.4f} ({arrow})  "
          f"folds={[round(s, 3) for s in rec['fold_scores']]}  {rec['minutes']:.1f} min")


# ------------------------------------------------------------------ readers
def load_logs(include_quick: bool = False) -> list[dict]:
    d = dirs()
    paths = sorted(d["logs"].glob("*.json"))
    if include_quick:
        paths += sorted(d["quick"].glob("log_*.json"))
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def load_oof(exp_id: str) -> pd.DataFrame:
    return read_preds(dirs()["oof"] / f"{exp_id}.csv")


def load_test_preds(exp_id: str) -> pd.DataFrame:
    return read_preds(dirs()["test"] / f"{exp_id}.csv")
