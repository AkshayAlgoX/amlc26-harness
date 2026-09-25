"""AMLC26 exp8 — C3 cross-fitted pruner OOF cache (one per candidate-pool version x view x pruner config).

Input: a retrieval-level node (c2_retrieval superset or v0_seed) + a retrieval view.
Model k trains on DEV (H1) rows with fold != k (exp7 fold assignment, shared with the matcher) and scores exactly
the rows of fold k — DEV out-of-fold and H2 alike. H3 is never in the store, so it can never be trained on.
Output node c3_pruner: prune.parquet (q, code, prune_p, prune_rank — aligned with the viewed pool order),
pruner_fold<k>.txt boosters, AUC per group, iterations, timing. Re-running with the same inputs is a cache hit.

  python -m experiments.agy2_exp8_pruner --src c2_retrieval/<key> [--neg-rate 0.2] [--params '{"max_rounds":300}']
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402

CODE_FILES = ["experiments/agy2_exp8_pruner.py", "experiments/agy2_exp8_features.py", "experiments/agy2_exp7_h3.py"]


def pruner_defaults() -> dict:
    p = dict(S.frozen_config()["pruner"])
    p.pop("features")
    return p


def prune_rank(q, code, p):
    """Rank within S1 by prune_p desc; ties by code asc (== exp7's stable sort of the (q, code)-ordered pool)."""
    o = np.lexsort((code, -p, q))
    qs = q[o]
    st = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    r = np.empty(len(q), np.int32)
    r[o] = np.arange(len(q)) - np.repeat(st, np.diff(np.r_[st, len(q)])) + 1
    return r


def run_pruner(store: S.Store, src_ref: str, view=None, params=None, neg_rate=1.0, features=None, seed=None,
               threads=None, deterministic=False) -> str:
    src = store.open(*src_ref.split("/"))
    if src.kind not in ("c2_retrieval", "v0_seed"):
        raise S.StoreError(f"pruner needs a retrieval-level node, got {src.kind}")
    view = F.view_key(view if view is not None else (src.man["params"].get("view") or F.EXP7_VIEW))
    prm = {**pruner_defaults(), **(params or {})}
    feats = list(features or F.PR_FEATS)
    seed = S.frozen_config()["seed"] if seed is None else seed
    node_params = {"view": view, "params": prm, "features": feats, "neg_rate": float(neg_rate), "seed": seed,
                   "fold": S.fold_cfg(), "deterministic": bool(deterministic)}
    code = S.code_sha(*CODE_FILES)
    key = S.node_key("c3_pruner", {"src": src.ref}, node_params, code)
    if store.exists("c3_pruner", key):
        F.log(f"cache hit c3_pruner/{key}")
        return f"c3_pruner/{key}"
    t0 = time.time()
    qt = src.table("queries")
    D = F.load_view(src, view)
    F.log(f"view loaded: {len(D):,} pairs ({time.time() - t0:.1f}s)")
    qv = D.q.values
    fold = qt.fold.values.astype(np.int8)[qv]
    es = qt.es.values.astype(bool)[qv]
    grp = qt.group.values[qv]
    is_dev = grp == "dev"
    w = store.create("c3_pruner", {"src": src.ref}, node_params, code, tags=["exp8"])
    try:
        t1 = time.time()
        pred, info = F.crossfit_lgb(D, feats, prm, fold, is_dev, es, S.fold_cfg()[1], seed, neg_rate, threads,
                                    model_dir=w.tmp, tag="pruner", deterministic=deterministic)
        fit_min = (time.time() - t1) / 60
        out = pd.DataFrame({"q": D.q.values, "code": D.code.values, "prune_p": pred.astype(np.float32),
                            "prune_rank": prune_rank(D.q.values, D.code.values, pred)})
        w.put_table("prune", out)
        info.pop("train_rows")
        w.extra = {**info, **F.auc_by_group(D.label.values, pred, grp), "rows": int(len(D)),
                   "load_sec": round(t1 - t0, 1), "fit_min": round(fit_min, 2), "total_min": round((time.time() - t0) / 60, 2)}
        node = w.finalize()
    except BaseException:
        w.abort()
        raise
    F.log(f"C3 {node.ref}: {json.dumps({k: v for k, v in node.man['extra'].items() if k != 'top_features'})}")
    return node.ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--view", default=None, help="JSON view (k_fwd, rev_k, cap, channels); default exp7 / seed view")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--neg-rate", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    ref = run_pruner(S.Store(), a.src, json.loads(a.view) if a.view else None, json.loads(a.params), a.neg_rate,
                     threads=a.threads)
    print(ref)


if __name__ == "__main__":
    main()
