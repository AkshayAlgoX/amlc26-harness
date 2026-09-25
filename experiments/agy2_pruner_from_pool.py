"""AMLC26 pruner re-training from the SAVED pre-budget pool (dev rows only; no retrieval recomputation; Kaggle).

Input: artifacts/c2c3_pool_prebudget.parquet from an h4/dev pipeline run (H4 rows carry label -1 and are dropped
here). Cross-fits the pruner on the exp7 folds with the frozen LightGBM parameters and reports, per variant, the
pruning loss at the Top-40 / Top-80 budgets: true links kept, pruned true links, and best-case macro F0.5.
Variants (pre-registered): base (frozen config, must reproduce the stored prune_p), cap1000 (max_rounds 1000).
Features added from C0 (e.g. B2 name evidence) need a pipeline change and a new freeze; they are not tried here.
env: POOL (parquet), CONFIG (frozen config json), OUT_DIR.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import agy2_exp8_store as S  # noqa: E402
from experiments.agy2_pipeline import crossfit, inner_es_flag  # noqa: E402

ENV = os.environ.get


def main():
    import lightgbm as lgb
    cfg = json.loads(Path(ENV("CONFIG", str(ROOT / "experiments/configs/h4_frozen.json"))).read_text())
    P = S.read_parquet(ENV("POOL"))
    P = P[(P.g == 0) & (P.label >= 0)].sort_values(["q", "code"], kind="stable").reset_index(drop=True)
    feats = cfg["pruner"]["features"]
    X = np.empty((len(P), len(feats)), np.float32)
    for j, f in enumerate(feats):
        X[:, j] = P[f].values
    y = P.label.values.astype(np.int8)
    es = inner_es_flag(P.s1_id.values, cfg["folds"]["salt"])
    q = P.q.values
    truth = P.groupby("q").label.sum()                      # links inside the pool; out-of-pool links are fixed
    res = {}
    for name, rounds in (("base", cfg["pruner"]["max_rounds"]), ("cap1000", 1000)):
        prm = cfg["pruner"]
        params = dict(objective="binary", learning_rate=prm["learning_rate"], num_leaves=prm["num_leaves"],
                      min_data_in_leaf=prm["min_data_in_leaf"], feature_fraction=prm["feature_fraction"],
                      bagging_fraction=prm["bagging_fraction"], bagging_freq=1, lambda_l2=prm["lambda_l2"],
                      max_bin=prm["max_bin"], seed=cfg["seed"], num_threads=os.cpu_count(), verbose=-1)
        iters = []

        def fit(tr, k):
            e = es[tr]
            dtr = lgb.Dataset(X[tr[~e]], y[tr[~e]])                     # identical to the pipeline's crossfit
            m = lgb.train(params, dtr, rounds, valid_sets=[lgb.Dataset(X[tr[e]], y[tr[e]], reference=dtr)],
                          callbacks=[lgb.early_stopping(prm["early_stopping"], verbose=False)])
            iters.append(m.best_iteration)
            return m
        pred = crossfit(P.fold.values, np.ones(len(P), bool), fit, lambda m, te: m.predict(X[te], num_iteration=m.best_iteration),
                        cfg["folds"]["nfold"])
        D = pd.DataFrame({"q": q, "p": pred, "label": y}).sort_values(["q", "p"], ascending=[True, False], kind="stable")
        D["r"] = D.groupby("q").cumcount() + 1
        out = {"best_iters": iters}
        if name == "base":
            out["max_abs_diff_vs_stored_prune_p"] = float(np.max(np.abs(pred - P.prune_p.values)))
        for top in (40, 80):
            kept = D[(D.r <= top) & (D.label == 1)].groupby("q").size().reindex(truth.index, fill_value=0)
            out[f"top{top}"] = {"true_links_kept": int(kept.sum()), "pruned_true_links": int(truth.sum() - kept.sum())}
        res[name] = out
        print(name, json.dumps(out), flush=True)
    o = Path(ENV("OUT_DIR", "/kaggle/working/pruner_pool"))
    o.mkdir(parents=True, exist_ok=True)
    (o / "pruner_from_pool.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
