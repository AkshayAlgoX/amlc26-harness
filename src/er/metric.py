"""Official AMLC26 evaluation metric: Macro F_0.5 Score per Source 1 entity.

Formula:
  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed as a macro-average across all Source 1 entities in the evaluation set.
Singletons:
  - true matches == empty and predicted matches == empty -> 1.0
  - true matches == empty and predicted matches != empty -> 0.0
Non-singletons:
  - true matches != empty and predicted matches == empty -> 0.0
  - TP == 0 -> 0.0
  - TP > 0 -> (1.25 * P * R) / (0.25 * P + R)
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np
import pandas as pd


def _to_set(val: Any) -> Set[str]:
    """Convert input into a set of clean string IDs."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, (set, frozenset)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, (list, tuple)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return set()
        return {part.strip() for part in s.split(",") if part.strip()}
    return {str(val).strip()}


def f05_score_single(true_set: Set[str], pred_set: Set[str]) -> float:
    """Compute F_0.5 for a single Source 1 entity.

    Official worked example:
      pred: [S2-00047, S2-00193, S3-00812]
      true: [S2-00047, S3-00812]
      P = 2/3, R = 1.0 -> F_0.5 = 5/7 ≈ 0.7142857
    """
    is_true_empty = len(true_set) == 0
    is_pred_empty = len(pred_set) == 0

    # Singleton case
    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0

    # Non-singleton case
    if is_pred_empty:
        return 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return float((1.25 * precision * recall) / denom)


def macro_f05(
    y_true: Union[Dict[str, Any], Iterable[Any]],
    y_pred: Union[Dict[str, Any], Iterable[Any]],
    s1_ids: Optional[Iterable[str]] = None,
) -> float:
    """Compute Macro F_0.5 score.

    Accepts:
      1) Two dicts: {s1_id: set_or_list_of_matched_ids}
      2) Two parallel iterables: ground truth and prediction values per S1 entity
    """
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        eval_ids = list(s1_ids) if s1_ids is not None else list(y_true.keys())
        if not eval_ids:
            return 0.0
        scores = [f05_score_single(_to_set(y_true.get(sid)), _to_set(y_pred.get(sid))) for sid in eval_ids]
        return float(np.mean(scores))

    # Parallel sequence input
    true_list = [_to_set(t) for t in y_true]
    pred_list = [_to_set(p) for p in y_pred]

    if len(true_list) != len(pred_list):
        raise ValueError(f"Length mismatch: {len(true_list)} true vs {len(pred_list)} pred.")

    if not true_list:
        return 0.0

    scores = [f05_score_single(t, p) for t, p in zip(true_list, pred_list)]
    return float(np.mean(scores))


def evaluate_matching(
    ground_truth: Dict[str, Any],
    predictions: Dict[str, Any],
    s1_ids: Optional[Iterable[str]] = None,
    countries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Detailed evaluation summary including singletons, non-singletons, and per-country metrics."""
    eval_ids = list(s1_ids) if s1_ids is not None else list(ground_truth.keys())
    if not eval_ids:
        return {"macro_f05": 0.0, "total_entities": 0}

    scores = []
    singleton_scores = []
    non_singleton_scores = []
    precisions = []
    recalls = []

    country_scores: Dict[str, List[float]] = {}

    for sid in eval_ids:
        t = _to_set(ground_truth.get(sid))
        p = _to_set(predictions.get(sid))
        s = f05_score_single(t, p)
        scores.append(s)

        if len(t) == 0:
            singleton_scores.append(s)
        else:
            non_singleton_scores.append(s)
            tp = len(t.intersection(p))
            prec = tp / len(p) if len(p) > 0 else 0.0
            rec = tp / len(t) if len(t) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        if countries and sid in countries:
            c = countries[sid]
            country_scores.setdefault(c, []).append(s)

    res: Dict[str, Any] = {
        "macro_f05": float(np.mean(scores)),
        "total_entities": len(eval_ids),
        "singleton_count": len(singleton_scores),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        "non_singleton_count": len(non_singleton_scores),
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
    }

    if country_scores:
        res["by_country"] = {c: float(np.mean(s_list)) for c, s_list in country_scores.items()}

    return res
