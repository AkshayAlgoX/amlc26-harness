"""Competition metrics. The ONLY place scores are computed.

DAY 1: open the problem PDF, find the metric definition, and make sure the matching
function below reproduces the official worked example exactly (add it to tests/test_metric.py).
Past editions:
  2023  mape_score     max(0, 100 * (1 - MAPE))            higher is better
  2024  f1_extraction  exact-match F1 with blank handling  higher is better
  2025  smape          mean |p-y| / ((|y|+|p|)/2) * 100    lower is better
"""
from __future__ import annotations

import numpy as np


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def smape(y_true, y_pred) -> float:
    """2025 definition. 0 when both are 0. Range 0..200, lower is better."""
    y, p = _f(y_true), _f(y_pred)
    denom = (np.abs(y) + np.abs(p)) / 2.0
    diff = np.abs(p - y)
    out = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return float(out.mean() * 100.0)


def mape_score(y_true, y_pred) -> float:
    """2023 definition: max(0, 100*(1-MAPE)), MAPE as in sklearn (eps-guarded). Higher is better."""
    y, p = _f(y_true), _f(y_pred)
    eps = np.finfo(np.float64).eps
    mape = np.mean(np.abs(p - y) / np.maximum(np.abs(y), eps))
    return float(max(0.0, 100.0 * (1.0 - mape)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_f(y_pred) - _f(y_true))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_f(y_pred) - _f(y_true)) ** 2)))


def rmsle(y_true, y_pred) -> float:
    y, p = np.clip(_f(y_true), 0, None), np.clip(_f(y_pred), 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def f1_extraction(y_true, y_pred) -> float:
    """2024-style exact-match F1 (as reproduced in participant repos — VERIFY vs the 2026 PDF).
      TP: out != "" and gt != "" and out == gt
      FP: out != "" and gt != "" and out != gt   OR   out != "" and gt == ""
      FN: out == "" and gt != ""
      TN: out == "" and gt == ""   (not counted)
    Predicting "" (abstaining) is a legal move: it costs recall but saves precision.
    """
    gt = [_s(g) for g in y_true]
    out = [_s(o) for o in y_pred]
    tp = fp = fn = 0
    for g, o in zip(gt, out):
        if o != "" and g != "" and o == g:
            tp += 1
        elif o != "":
            fp += 1
        elif g != "":
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str), average="macro"))


# name -> (function, greater_is_better)
METRICS = {
    "smape": (smape, False),
    "mape_score": (mape_score, True),
    "mae": (mae, False),
    "rmse": (rmse, False),
    "rmsle": (rmsle, False),
    "f1_extraction": (f1_extraction, True),
    "accuracy": (accuracy, True),
    "macro_f1": (macro_f1, True),
}


def _name(name: str | None) -> str:
    if name is None:
        from .config import task

        name = task()["metric"]
    if name not in METRICS:
        raise KeyError(f"Unknown metric '{name}'. Known: {sorted(METRICS)}")
    return name


def score(y_true, y_pred, name: str | None = None) -> float:
    """Score with the configured competition metric (or an explicit one)."""
    return METRICS[_name(name)][0](y_true, y_pred)


def greater_is_better(name: str | None = None) -> bool:
    return METRICS[_name(name)][1]


def is_better(a: float, b: float, name: str | None = None) -> bool:
    """True if score a is strictly better than score b under the metric's direction."""
    return a > b if greater_is_better(name) else a < b
