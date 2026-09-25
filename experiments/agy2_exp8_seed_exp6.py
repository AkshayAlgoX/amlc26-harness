"""AMLC26 exp8 — v0 sandbox: seed the store from the existing exp6 pairs_pool.parquet (no raw-data scan).

The exp6 pool (23.6M exact unique (S1, target) pairs, H1 + H2 queries only, no H3) is copied byte-for-byte into a
v0_seed node together with a queries table:
  q -> S1 id   : exp6 built Q = prev_val_s1_50k.ids (H1, q < 50k) + exp5_holdout_s1.ids (H2) in file order,
  fold / es    : the exp7 frozen fold assignment (pure function of the S1 id), shared by pruner and matcher,
  truth_n = -1 : per-S1 ground-truth counts are NOT recoverable from the pool; pool_hits (links present in the pool)
                 is stored instead and C6 labels its scores "pool_truth_proxy".
Tag exp6-legacy: built with the legacy plain tab-split parser and statistics that excluded H2 only (exp7 excludes
H2 + H3); flags are exp6's fixed view (F@200, R@5, cap 200). No text sidecar -> no Jaro-Winkler features.
Use it for plumbing, timing and relative set-decision / pruner / matcher-hyperparameter experiments — never for a
reported score. The real snapshot is agy2_exp8_build_snapshot.py.

  EXP8_IDS_ROOT=<dir with exclude_ids/ expected/> python -m experiments.agy2_exp8_seed_exp6 --pool <pairs_pool.parquet>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_features as F  # noqa: E402
from experiments import agy2_exp8_store as S  # noqa: E402

CODE_FILES = ["experiments/agy2_exp8_seed_exp6.py"]
EXP6_VIEW = {"k_fwd": 200, "rev_k": 5, "cap": 200, "channels": list(F.CHANNELS)}
EXP6_COLS = ["q", "code", "label", "grp", "is_s2"] + [c for c in F.PR_FEATS]


def seed_exp6(store: S.Store, pool_path) -> tuple:
    t0 = time.time()
    pool_path = Path(pool_path)
    dev, h2 = S.group_ids("dev"), S.group_ids("h2")
    Q = dev + h2
    pool_sha = S.sha256_file(pool_path)
    t_sha = time.time() - t0
    params = {"source": "exp6 pairs_pool.parquet", "pool_sha256": pool_sha, "view": F.view_key(EXP6_VIEW),
              "groups": {"dev": S.frozen_config()["groups"]["dev"]["sha256"], "h2": S.frozen_config()["groups"]["h2"]["sha256"]},
              "fold": S.fold_cfg()}
    code = S.code_sha(*CODE_FILES)
    key = S.node_key("v0_seed", {}, params, code)
    if store.exists("v0_seed", key):
        store.open("v0_seed", key)
        return f"v0_seed/{key}", {"cache_hit": True, "total_sec": round(time.time() - t0, 2)}
    cols = S.parquet_columns(pool_path)
    missing = [c for c in EXP6_COLS if c not in cols]
    if missing:
        raise S.StoreError(f"exp6 pool lacks columns {missing}")
    t1 = time.time()
    P = S.read_parquet(pool_path, columns=["q", "grp", "label", "country"])
    t_read = time.time() - t1
    q = P.q.values.astype(np.int64)
    if q.min() < 0 or q.max() >= len(Q):
        raise S.StoreError("exp6 q outside H1+H2")
    grp = np.asarray(P.grp.values).astype(str)
    if not np.array_equal(grp == "train", q < len(dev)) or not set(np.unique(grp)) <= {"train", "fresh"}:
        raise S.StoreError("exp6 grp does not match q < |H1| (id order assumption broken)")
    if (np.diff(q) < 0).any():
        raise S.StoreError("exp6 pool not sorted by q")
    hits = np.bincount(q[P.label.values.astype(bool)], minlength=len(Q))
    first = np.r_[0, np.flatnonzero(np.diff(q)) + 1]
    cty = np.full(len(Q), -1, np.int8)
    cty[q[first]] = P.country.values[first]
    n_pairs, n_pos = len(P), int(hits.sum())
    del P
    fold, es = S.folds_for(Q)
    queries = pd.DataFrame({"q": np.arange(len(Q), dtype=np.int32), "s1_id": Q,
                            "group": ["dev"] * len(dev) + ["h2"] * len(h2), "fold": fold, "es": es,
                            "truth_n": np.full(len(Q), -1, np.int32), "pool_hits": hits.astype(np.int32), "country": cty})
    w = store.create("v0_seed", {}, params, code, tags=["exp6-legacy", "v0-sandbox"])
    t2 = time.time()
    w.put_table("queries", queries)
    w.put_file("pool.parquet", src=pool_path)
    t_copy = time.time() - t2
    w.extra = {"pairs": n_pairs, "positives_in_pool": n_pos, "truth_complete": False,
               "contract": "exp6-legacy: plain tab-split parser; text statistics excluded H2 only; fixed F@200/R@5/cap200 view",
               "exp6_links_total": {"dev": 172636, "h2": 173103}}
    t3 = time.time()
    node = w.finalize()
    timing = {"sha_input_sec": round(t_sha, 2), "read_validate_sec": round(t_read, 2), "copy_sec": round(t_copy, 2),
              "finalize_hash_sec": round(time.time() - t3, 2), "total_sec": round(time.time() - t0, 2)}
    F.log(f"v0 seeded {node.ref}: {n_pairs:,} pairs, {n_pos:,} positives | {timing}")
    return node.ref, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    a = ap.parse_args()
    ref, timing = seed_exp6(S.Store(), a.pool)
    print(ref, timing)


if __name__ == "__main__":
    main()
