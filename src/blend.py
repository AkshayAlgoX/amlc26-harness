"""Blend experiments using ONLY their OOF predictions, then apply the same recipe to test.

  python -m src.blend --top 8                  # best 8 full runs by CV
  python -m src.blend --exps a b c --name cap-blend-004
  python -m src.blend --top 8 --space linear --no-calibrate

Numeric targets: greedy forward selection with replacement (Caruana-style hill climbing)
over a weighted mean, in log space by default for percentage metrics, followed by a global
multiplier calibrated on OOF. Label/string targets: weighted vote, same selection loop.
The blend is saved like any experiment (oof/, test/, logs/), so it shows up on the board
and can be submitted with `python -m src.make_submission --exp <name>`.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .config import col, task
from .data import load_train
from .metric import greater_is_better, is_better, score
from .registry import _print, _record, _write, dirs, load_logs, load_oof, load_test_preds

LOG_METRICS = {"smape", "mape_score", "rmsle"}


def _combine_numeric(mat: np.ndarray, w: np.ndarray, log_space: bool) -> np.ndarray:
    w = w / w.sum()
    if log_space:
        return np.expm1(np.log1p(np.clip(mat, 0, None)) @ w)
    return mat @ w


def _combine_vote(codes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """codes: N x K int matrix (columns ordered best model first). Returns winning code per row."""
    n, k = codes.shape
    support = np.zeros((n, k))
    for j in range(k):
        support[:, j] = ((codes == codes[:, [j]]) * w).sum(axis=1)
    return codes[np.arange(n), support.argmax(axis=1)]  # argmax -> first max -> best model wins ties


def hill_climb(eval_fn, k: int, single_scores: list[float], max_rounds: int = 60) -> tuple[np.ndarray, float]:
    """Greedy forward selection with replacement. eval_fn(counts)->score."""
    order = sorted(range(k), key=lambda i: single_scores[i], reverse=greater_is_better())
    counts = np.zeros(k)
    counts[order[0]] = 1
    cur = eval_fn(counts)
    for _ in range(max_rounds):
        best_j, best_s = None, None
        for j in range(k):
            counts[j] += 1
            s = eval_fn(counts)
            counts[j] -= 1
            if best_s is None or is_better(s, best_s):
                best_j, best_s = j, s
        if best_s is None or not is_better(best_s, cur):
            break
        counts[best_j] += 1
        cur = best_s
    return counts / counts.sum(), cur


def calibrate(y: np.ndarray, p: np.ndarray, lo: float = 0.8, hi: float = 1.2, steps: int = 81) -> tuple[float, float]:
    grid = np.linspace(lo, hi, steps)
    scores = [score(y, p * m) for m in grid]
    best = int(np.argmax(scores) if greater_is_better() else np.argmin(scores))
    return float(grid[best]), float(scores[best])


def _next_name() -> str:
    nums = [int(m.group(1)) for p in dirs()["logs"].glob("cap-blend-*.json")
            if (m := re.match(r"cap-blend-(\d{3})", p.stem))]
    return f"cap-blend-{(max(nums) + 1) if nums else 1:03d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--space", choices=["auto", "log", "linear"], default="auto")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--rounds", type=int, default=60)
    args = ap.parse_args(argv)

    logs = [r for r in load_logs() if r["kind"] == "model"]
    if args.exps:
        known = {r["exp_id"] for r in logs}
        missing = [e for e in args.exps if e not in known]
        if missing:
            raise SystemExit(f"not found / not full model runs: {missing}")
        names = args.exps
    else:
        logs.sort(key=lambda r: r["cv"], reverse=greater_is_better())
        names = [r["exp_id"] for r in logs[: args.top]]
    if len(names) < 2:
        raise SystemExit("need at least 2 full model runs to blend")

    tr = load_train()
    truth = pd.DataFrame({"id": tr[col("id")].astype(str), "y": tr[col("target")]})
    base = truth[["id"]]
    oofs, tests = [], []
    for n in names:
        o = load_oof(n)
        if set(o["id"]) != set(base["id"]):
            raise SystemExit(f"{n}: OOF ids differ from train ids — was it a QUICK run or a different folds.csv?")
        oofs.append(base.merge(o[["id", "pred"]], on="id", how="left")["pred"].to_numpy())
        tests.append(load_test_preds(n))
    test_ids = tests[0]["id"]
    for n, t in zip(names, tests):
        if set(t["id"]) != set(test_ids):
            raise SystemExit(f"{n}: test ids differ from {names[0]} — mixed QUICK/full runs?")
    tests = [t.set_index("id").loc[test_ids, "pred"].to_numpy() for t in tests]
    y = truth["y"].to_numpy()
    singles = [score(y, p) for p in oofs]
    numeric = task()["type"] == "regression"

    extra = {"members": names}
    if numeric:
        mat, tmat = np.column_stack(oofs).astype(float), np.column_stack(tests).astype(float)
        log_space = (args.space == "log") or (args.space == "auto" and task()["metric"] in LOG_METRICS
                                               and (mat > 0).all() and (tmat > 0).all())
        weights, cv = hill_climb(lambda c: score(y, _combine_numeric(mat, c, log_space)), len(names), singles, args.rounds)
        oof_blend = _combine_numeric(mat, weights, log_space)
        test_blend = _combine_numeric(tmat, weights, log_space)
        mult = 1.0
        if not args.no_calibrate:
            mult, cv_cal = calibrate(y, oof_blend)
            if is_better(cv_cal, cv):
                oof_blend, test_blend, cv = oof_blend * mult, test_blend * mult, cv_cal
            else:
                mult = 1.0
        extra.update({"space": "log" if log_space else "linear", "multiplier": mult})
    else:
        order = sorted(range(len(names)), key=lambda i: singles[i], reverse=greater_is_better())
        allv = np.concatenate([np.column_stack(oofs).ravel(), np.column_stack(tests).ravel()]).astype(str)
        codes_all, uniques = pd.factorize(allv)
        n_o = len(y) * len(names)
        codes = codes_all[:n_o].reshape(len(y), len(names))[:, order]
        tcodes = codes_all[n_o:].reshape(len(test_ids), len(names))[:, order]
        names = [names[i] for i in order]
        singles = [singles[i] for i in order]
        weights, cv = hill_climb(lambda c: score(y, uniques[_combine_vote(codes, c)]), len(names), singles, args.rounds)
        oof_blend = np.asarray(uniques[_combine_vote(codes, weights)], dtype=object)
        test_blend = np.asarray(uniques[_combine_vote(tcodes, weights)], dtype=object)
        extra["members"] = names

    extra["weights"] = {n: round(float(w), 4) for n, w in zip(names, weights) if w > 0}
    extra["single_cv"] = {n: round(float(s), 6) for n, s in zip(names, singles)}
    name = args.name or _next_name()
    oof_df = pd.DataFrame({"id": base["id"], "pred": oof_blend})
    oof_df = oof_df.merge(load_oof(names[0])[["id", "fold"]], on="id")[["id", "fold", "pred"]]
    test_df = pd.DataFrame({"id": test_ids.astype(str), "pred": test_blend})
    m = oof_df.merge(truth, on="id")
    per_fold = [score(g["y"].to_numpy(), g["pred"].to_numpy()) for _, g in m.groupby("fold")]
    rec = _record(name, "blend", oof_df, per_fold, cv, 0.0, f"blend of {len(extra['weights'])} runs", False,
                  len(test_df), extra)
    _write(name, oof_df, test_df, rec, False)
    print("single-model CV:", extra["single_cv"])
    print("weights:", extra["weights"], "| space:", extra.get("space", "vote"), "| multiplier:", extra.get("multiplier", "-"))
    _print(rec)


if __name__ == "__main__":
    main()
