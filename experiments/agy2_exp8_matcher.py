"""AMLC26 exp8 — C4 matcher feature cache + C5 cross-fitted matcher OOF scores.

C4 (c4_mfeat): viewed pool + cached C3 prune_p/prune_rank -> budget (top_n, floor) -> expensive text features
     (Jaro-Winkler, containment, ...) computed ONLY on the pruned budget, in parallel chunks. Cached per
     (C3 node, budget, feature code). Nodes without text sidecars (v0_seed) get the non-text features only.
C5 (c5_matcher): LightGBM on a C4 table with the SAME fold per S1 as the pruner: matcher k trains on DEV rows with
     fold != k, whose prune_p came from pruner_{fold(row)} (never trained on that row's S1); it scores fold k, DEV-OOF
     and H2 alike, so every row's score comes from the single chain pruner_k -> matcher_k of its own fold.

  python -m experiments.agy2_exp8_matcher feats   --c3 c3_pruner/<key> [--top-n 80 --floor 0]
  python -m experiments.agy2_exp8_matcher train   --c4 c4_mfeat/<key>  [--neg-rate 0.3 --features a,b,c --params '{}']
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402
from experiments import agy2_exp8_build_snapshot as B  # noqa: E402

FEAT_CODE = ["experiments/agy2_exp8_matcher.py", "experiments/agy2_exp8_features.py", "experiments/agy2_exp6_integrated.py"]
TRAIN_CODE = ["experiments/agy2_exp8_matcher.py", "experiments/agy2_exp8_features.py", "experiments/agy2_exp7_h3.py"]


def matcher_defaults() -> dict:
    p = dict(S.frozen_config()["matcher"])
    p.pop("features")
    return p


def _text_chunk(rng):
    lo, hi = rng
    G = B._G
    sub = G["M"].iloc[lo:hi]
    return F.text_pair_features(sub, G["qname"], G["qaddr"], G["tname"], G["taddr"])


def budget_table(src, c3, top_n, floor) -> pd.DataFrame:
    D = F.load_view(src, c3.man["params"]["view"])
    pr = c3.table("prune")
    if not (np.array_equal(pr.q.values, D.q.values) and np.array_equal(pr.code.values, D.code.values)):
        raise S.StoreError("C3 prune table is not aligned with the viewed pool — view/pool changed")
    D["prune_p"], D["prune_rank"] = pr.prune_p.values, pr.prune_rank.values
    return D[(D.prune_rank.values <= top_n) & (D.prune_p.values >= floor)].reset_index(drop=True)


def run_mfeat(store: S.Store, c3_ref: str, top_n=None, floor=None, workers=None) -> str:
    c3 = store.open(*c3_ref.split("/"))
    src = c3.source()
    bud = S.frozen_config()["budget"]
    top_n = int(bud["top_n"] if top_n is None else top_n)
    floor = float(bud["floor"] if floor is None else floor)
    has_text = "t_text.parquet" in src.man["files"]
    params = {"top_n": top_n, "floor": floor, "text_features": F.TEXT_FEATS if has_text else []}
    code = S.code_sha(*FEAT_CODE)
    key = S.node_key("c4_mfeat", {"c3": c3.ref}, params, code)
    if store.exists("c4_mfeat", key):
        F.log(f"cache hit c4_mfeat/{key}")
        return f"c4_mfeat/{key}"
    t0 = time.time()
    M = budget_table(src, c3, top_n, floor)
    F.log(f"budget top{top_n}_floor{floor}: {len(M):,} pairs")
    if has_text:
        qt, tt = src.table("q_text"), src.table("t_text")
        workers = workers or os.cpu_count()
        B._G.update(M=M[["q", "t_idx", "exact_name", "exact_addr"]], qname=qt.name.tolist(), qaddr=qt.addr.tolist(),
                    tname=tt.name.tolist(), taddr=tt.addr.tolist())
        parts = B.pmap(_text_chunk, B.ranges(len(M), workers * 4), workers)
        B._G.clear()
        for col in parts[0]:
            M[col] = np.concatenate([p[col] for p in parts])
    w = store.create("c4_mfeat", {"c3": c3.ref}, params, code, tags=["exp8"])
    w.put_table("mfeat", M)
    w.extra = {"pairs": int(len(M)), "has_text_features": has_text, "build_min": round((time.time() - t0) / 60, 2)}
    node = w.finalize()
    F.log(f"C4 {node.ref}: {node.man['extra']}")
    return node.ref


def run_matcher(store: S.Store, c4_ref: str, params=None, features=None, neg_rate=1.0, seed=None, threads=None,
                deterministic=False) -> str:
    c4 = store.open(*c4_ref.split("/"))
    c3 = c4.parent("c3")
    src = c4.source()
    if list(c3.man["params"]["fold"]) != list(S.fold_cfg()):
        raise S.StoreError("pruner fold assignment differs from the matcher's")
    cols = S.parquet_columns(c4.file("mfeat.parquet"))
    feats = list(features or [f for f in F.M_FEATS if f in cols])
    missing = [f for f in feats if f not in cols]
    if missing:
        raise ValueError(f"features not in C4 table: {missing}")
    prm = {**matcher_defaults(), **(params or {})}
    seed = S.frozen_config()["seed"] if seed is None else seed
    node_params = {"params": prm, "features": feats, "neg_rate": float(neg_rate), "seed": seed, "fold": S.fold_cfg(),
                   "deterministic": bool(deterministic)}
    code = S.code_sha(*TRAIN_CODE)
    key = S.node_key("c5_matcher", {"c4": c4.ref}, node_params, code)
    if store.exists("c5_matcher", key):
        F.log(f"cache hit c5_matcher/{key}")
        return f"c5_matcher/{key}"
    t0 = time.time()
    M = c4.table("mfeat", columns=sorted(set(["q", "code", "label"] + feats)))
    qt = src.table("queries")
    qv = M.q.values
    fold, es, grp = qt.fold.values.astype(np.int8)[qv], qt.es.values.astype(bool)[qv], qt.group.values[qv]
    w = store.create("c5_matcher", {"c4": c4.ref}, node_params, code, tags=["exp8"])
    try:
        t1 = time.time()
        p, info = F.crossfit_lgb(M, feats, prm, fold, grp == "dev", es, S.fold_cfg()[1], seed, neg_rate, threads,
                                 model_dir=w.tmp, tag="matcher", deterministic=deterministic)
        info.pop("train_rows")
        w.put_table("scores", pd.DataFrame({"q": M.q.values, "code": M.code.values, "p": p.astype(np.float32),
                                            "label": M.label.values}))
        w.extra = {**info, **F.auc_by_group(M.label.values, p, grp), "rows": int(len(M)),
                   "load_sec": round(t1 - t0, 1), "fit_min": round((time.time() - t1) / 60, 2)}
        node = w.finalize()
    except BaseException:
        w.abort()
        raise
    F.log(f"C5 {node.ref}: {json.dumps({k: v for k, v in node.man['extra'].items() if k != 'top_features'})}")
    return node.ref


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("feats"); a1.add_argument("--c3", required=True)
    a1.add_argument("--top-n", type=int); a1.add_argument("--floor", type=float); a1.add_argument("--workers", type=int)
    a2 = sp.add_parser("train"); a2.add_argument("--c4", required=True); a2.add_argument("--params", default="{}")
    a2.add_argument("--features", default=None); a2.add_argument("--neg-rate", type=float, default=1.0)
    a2.add_argument("--threads", type=int)
    a = ap.parse_args()
    if a.cmd == "feats":
        print(run_mfeat(S.Store(), a.c3, a.top_n, a.floor, a.workers))
    else:
        print(run_matcher(S.Store(), a.c4, json.loads(a.params), a.features.split(",") if a.features else None,
                          a.neg_rate, threads=a.threads))


if __name__ == "__main__":
    main()
