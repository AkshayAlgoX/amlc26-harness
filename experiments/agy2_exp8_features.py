"""AMLC26 exp8 — shared pair features, retrieval views and the cross-fitted LightGBM trainer.

Retrieval VIEW: the C2 pool stores a SUPERSET (forward rank <= 300, reverse rank <= 10, capped-channel effective
bucket counts <= 1000). A view (k_fwd, rev_k, cap, channels) turns it back into exactly the pool + flags an exp7-style
run with those settings would produce, and recomputes the per-S1 aggregates (cands_s1, fwd_ratio_best,
name_jacc_rank) on the viewed pool. The exp7 frozen contract is EXP7_VIEW.

Cross-fitting: ONE fold per S1 (exp7 fold_of). Model k trains only on DEV rows with fold != k and scores exactly
the rows with fold == k (DEV-OOF and every evaluation group), so an evaluation S1 is always scored by the single
chain of its own fold. Optional deterministic negative subsampling (positives always kept) with prior correction.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.agy2_exp7_h3 import M_FEATS, PR_FEATS  # noqa: E402  (frozen feature lists, imported read-only)

CHANNELS = ("A", "B", "F200", "R5", "B2C", "B3C", "B4C")
EXP7_VIEW = {"k_fwd": 200, "rev_k": 5, "cap": 200, "channels": list(CHANNELS)}
SUPERSET = {"k_fwd": 300, "rev_k": 10, "cap": 1000}
FWD_NONE, REV_NONE, EFF_NONE = 9999, 99, np.iinfo(np.int32).max
TEXT_FEATS = [f for f in M_FEATS if f not in PR_FEATS and f not in ("prune_p", "prune_rank")]
SUPERSET_COLS = ["q", "code", "label", "is_s2", "flag_A", "flag_B", "fwd_rank", "fwd_score", "rev_rank", "rev_score",
                 "b2_eff", "b3_eff", "b4_eff", "name_jacc", "name_cont", "addr_jacc", "digit_jacc", "exact_name",
                 "exact_addr", "num_match", "pin_match", "q_addr_missing", "t_addr_missing", "country", "src_s3",
                 "name_len_diff", "t_idx"]


def log(msg, t0=[time.time()]):
    print(f"[{(time.time() - t0[0]) / 60:6.2f} min] {msg}", flush=True)


def view_key(view: dict) -> dict:
    v = {**EXP7_VIEW, **(view or {})}
    v["channels"] = sorted(v["channels"])
    return v


# ------------------------------------------------------------------ retrieval view over a superset pool
def apply_view(pool: pd.DataFrame, view: dict, superset: dict) -> pd.DataFrame:
    """Superset pool -> exp7-style pool (membership, flags, scores, per-S1 aggregates) for the given view."""
    v = view_key(view)
    k, rk, cap = v["k_fwd"], v["rev_k"], v["cap"]
    if k > superset["k_fwd"] or rk > superset["rev_k"] or cap > superset["cap"]:
        raise ValueError(f"view {v} exceeds the stored superset {superset}")
    fr, rr = pool.fwd_rank.values, pool.rev_rank.values
    f200 = fr <= k
    r5 = rr <= rk
    ch = {"A": pool.flag_A.values > 0, "B": pool.flag_B.values > 0, "F200": f200, "R5": r5,
          "B2C": pool.b2_eff.values <= cap, "B3C": pool.b3_eff.values <= cap, "B4C": pool.b4_eff.values <= cap}
    keep = np.zeros(len(pool), bool)
    for c in v["channels"]:
        keep |= ch[c]
    D = pool.loc[keep].copy()
    m = keep
    D["flag_F100"] = (fr[m] <= min(100, k)).astype(np.int8)
    D["flag_F200"] = f200[m].astype(np.int8)
    D["flag_R1"] = (rr[m] == 1).astype(np.int8) if rk >= 1 else np.zeros(m.sum(), np.int8)
    D["flag_R5"] = r5[m].astype(np.int8)
    D["flag_b2"], D["flag_b3"], D["flag_b4"] = (ch["B2C"][m].astype(np.int8), ch["B3C"][m].astype(np.int8),
                                                ch["B4C"][m].astype(np.int8))
    D["flag_A"], D["flag_B"] = D.flag_A.astype(np.int8), D.flag_B.astype(np.int8)
    D["num_channels"] = (D.flag_A + D.flag_B + D.flag_F200 + D.flag_R5 + D.flag_b2 + D.flag_b3 + D.flag_b4).astype(np.int8)
    D["fwd_tfidf"] = np.where(f200[m], D.fwd_score.values, 0.0).astype(np.float32)
    D["fwd_rank"] = np.where(f200[m], fr[m], FWD_NONE).astype(np.int32)
    D["rev_tfidf"] = np.where(r5[m], D.rev_score.values, 0.0).astype(np.float32)
    D["rev_rank"] = np.where(r5[m], rr[m], REV_NONE).astype(np.int16)
    return add_group_aggregates(D.reset_index(drop=True))


def add_group_aggregates(D: pd.DataFrame) -> pd.DataFrame:
    g = D.groupby("q", sort=False)
    D["cands_s1"] = g.q.transform("size").astype(np.int32)
    D["fwd_ratio_best"] = (D.fwd_tfidf / g.fwd_tfidf.transform("max").replace(0, np.nan)).fillna(0).astype(np.float32)
    D["name_jacc_rank"] = g.name_jacc.rank(ascending=False, method="min").astype(np.float32)
    return D


def load_view(src, view: dict) -> pd.DataFrame:
    """Viewed pool of a retrieval-level node (c2_retrieval superset, or the fixed-view v0 seed)."""
    if src.kind == "v0_seed":
        if view_key(view) != view_key(src.man["params"]["view"]):
            raise ValueError("v0_seed is a fixed exp6 view; only its own view can be used")
        from experiments.agy2_exp8_store import parquet_columns
        return src.table("pool", columns=[c for c in parquet_columns(src.file("pool.parquet")) if c != "grp"])
    return apply_view(src.table("pool"), view, src.man["params"]["superset"])


# ------------------------------------------------------------------ cheap pair features (exp7 definitions)
def cheap_pair_features(qv, tv, qname, qaddr, q_parts, tname, taddr, tpin, tnum):
    """Per-pair cheap features exactly as exp7 lines 418-442 (token Jaccard/containment, exact, number/pin match)."""
    n = len(qv)
    f_nj, f_nc, f_aj, f_dj = (np.zeros(n, np.float32) for _ in range(4))
    f_en, f_ea, f_num, f_pin = (np.zeros(n, np.int8) for _ in range(4))
    qn_tok, qa_tok, qa_dig = {}, {}, {}
    for r in range(n):
        i, j = int(qv[r]), int(tv[r])
        if i not in qn_tok:
            qn_tok[i] = set(qname[i].split()); qa_tok[i] = set(qaddr[i].split())
            qa_dig[i] = {t for t in qaddr[i].split() if t.isdigit()}
        tn_, ta_ = tname[j], taddr[j]
        a, b = qn_tok[i], set(tn_.split())
        inter = len(a & b)
        f_nj[r] = inter / max(len(a | b), 1); f_nc[r] = inter / max(min(len(a), len(b)), 1)
        c_, d_ = qa_tok[i], set(ta_.split())
        f_aj[r] = len(c_ & d_) / max(len(c_ | d_), 1)
        dd = {t for t in d_ if t.isdigit()}
        f_dj[r] = len(qa_dig[i] & dd) / max(len(qa_dig[i] | dd), 1) if (qa_dig[i] or dd) else 0.0
        f_en[r] = qname[i] == tn_ and len(tn_) > 0
        f_ea[r] = qaddr[i] == ta_ and len(ta_) > 0
        qp, qn2 = q_parts[i]
        f_num[r] = bool(qn2) and qn2 == tnum[j]
        f_pin[r] = bool(qp) and qp == tpin[j]
    return {"name_jacc": f_nj, "name_cont": f_nc, "addr_jacc": f_aj, "digit_jacc": f_dj,
            "exact_name": f_en, "exact_addr": f_ea, "num_match": f_num, "pin_match": f_pin}


def text_pair_features(M: pd.DataFrame, qname, qaddr, tname, taddr):
    """Expensive matcher features (exp7 lines 509-522), computed only on the pruned budget."""
    from experiments.agy2_exp6_integrated import jaro_winkler_pairs
    tvm, qvm = M.t_idx.values, M.q.values
    out = {}
    out["name_jw"] = jaro_winkler_pairs(qname, qvm, tname, tvm)
    out["addr_jw"] = jaro_winkler_pairs(qaddr, qvm, taddr, tvm)
    qn_tok = {i: set(qname[i].split()) for i in np.unique(qvm)}
    qa_tok = {i: set(qaddr[i].split()) for i in np.unique(qvm)}
    ql = np.array([len(qn_tok[i]) for i in qvm]); tl = np.array([len(tname[j].split()) for j in tvm])
    inter = np.array([len(qn_tok[i] & set(tname[j].split())) for i, j in zip(qvm, tvm)])
    out["name_cont_q"] = inter / np.maximum(ql, 1); out["name_cont_t"] = inter / np.maximum(tl, 1)
    q_nlen = np.array([len(qname[i]) for i in qvm]); t_nlen = np.array([len(tname[j]) for j in tvm])
    out["name_len_ratio"] = (q_nlen + 1) / (t_nlen + 1)
    out["tok_count_diff"] = np.abs(ql - tl)
    ainter = np.array([len(qa_tok[i] & set(taddr[j].split())) for i, j in zip(qvm, tvm)])
    out["addr_cont"] = ainter / np.maximum(np.array([len(qa_tok[i]) for i in qvm]), 1)
    out["exact_both"] = (M.exact_name.values & M.exact_addr.values).astype(np.int8)
    nj, aj = out["name_jw"], out["addr_jw"]
    out["strong_n_strong_a"] = ((nj >= 0.9) & (aj >= 0.9)).astype(np.int8)
    out["strong_n_weak_a"] = ((nj >= 0.9) & (aj < 0.5)).astype(np.int8)
    out["weak_n_strong_a"] = ((nj < 0.5) & (aj >= 0.9)).astype(np.int8)
    out["name_x_addr"] = nj * aj
    return out


# ------------------------------------------------------------------ cross-fitted LightGBM
def _unif(q, code, seed):
    """Deterministic U[0,1) per pair (splitmix64 of the pair key) — negative subsampling independent of row order."""
    x = ((q.astype(np.uint64) << np.uint64(32)) | code.astype(np.uint64)) ^ np.uint64(seed * 0x9E3779B97F4A7C15 % (1 << 64))
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _matrix(D, feats, rows=None):
    n = len(D) if rows is None else len(rows)
    X = np.empty((n, len(feats)), dtype=np.float32)          # column-wise fill: no float64 intermediate
    for j, f_ in enumerate(feats):
        v = D[f_].values
        X[:, j] = v if rows is None else v[rows]
    return X


def crossfit_lgb(D: pd.DataFrame, feats, prm: dict, fold, is_dev, es, nfold: int, seed: int, neg_rate: float = 1.0,
                 threads=None, model_dir=None, tag="model", deterministic=False, pred_chunk=2_000_000):
    """Returns (pred, info). Model k: trained on is_dev & fold != k (negatives kept with prob neg_rate),
    early-stopped on its own inner es subset, predicts exactly fold == k. info['train_rows'][k] lists row ids used.
    Memory: a training matrix per fold and chunked prediction — never a full float matrix of D."""
    import lightgbm as lgb
    y = D.label.values.astype(np.int8)
    keep_neg = np.ones(len(D), bool) if neg_rate >= 1 else (y == 1) | (_unif(D.q.values, D.code.values, seed) < neg_rate)
    params = dict(objective="binary", learning_rate=prm["learning_rate"], num_leaves=prm["num_leaves"],
                  min_data_in_leaf=prm["min_data_in_leaf"], feature_fraction=prm["feature_fraction"],
                  bagging_fraction=prm["bagging_fraction"], bagging_freq=1, lambda_l2=prm["lambda_l2"],
                  max_bin=prm["max_bin"], seed=seed, num_threads=threads or os.cpu_count(), verbose=-1)
    if deterministic:
        params.update(deterministic=True, force_row_wise=True)
    pred = np.full(len(D), np.nan)
    info = {"best_iters": [], "train_rows": {}, "n_train": [], "fit_sec": []}
    imps = np.zeros(len(feats))
    for k in range(nfold):
        t = time.time()
        tr = np.flatnonzero(is_dev & (fold != k) & keep_neg)
        te = np.flatnonzero(fold == k)
        assert not (fold[tr] == k).any() and is_dev[tr].all()
        e = es[tr]
        Xtr = _matrix(D, feats, tr)
        dtr = lgb.Dataset(Xtr[~e], y[tr[~e]], free_raw_data=True)
        dva = lgb.Dataset(Xtr[e], y[tr[e]], reference=dtr)
        m = lgb.train(params, dtr, prm["max_rounds"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(prm["early_stopping"], verbose=False)])
        del Xtr, dtr, dva
        for s0 in range(0, len(te), pred_chunk):
            rows = te[s0:s0 + pred_chunk]
            p = m.predict(_matrix(D, feats, rows), num_iteration=m.best_iteration)
            if neg_rate < 1:                                  # prior correction back to the full-data odds
                p = neg_rate * p / (neg_rate * p + (1 - p))
            pred[rows] = p
        if model_dir is not None:
            m.save_model(str(Path(model_dir) / f"{tag}_fold{k}.txt"), num_iteration=m.best_iteration)
        imps += m.feature_importance("gain")
        info["best_iters"].append(int(m.best_iteration)); info["n_train"].append(int(len(tr)))
        info["train_rows"][k] = tr
        info["fit_sec"].append(round(time.time() - t, 2))
        log(f"{tag} fold {k}: {len(tr):,} DEV rows (neg_rate {neg_rate}), best_iter {m.best_iteration}, {time.time() - t:.1f}s")
    assert not np.isnan(pred).any(), "crossfit left rows unscored"
    info["top_features"] = [f for _, f in sorted(zip(imps, feats), reverse=True)[:12]]
    return pred, info


def predict_by_fold(model_dir, tag, D, feats, fold, nfold, neg_rate):
    """Score rows with the saved model of their own fold (sealed holdout / test path)."""
    import lightgbm as lgb
    pred = np.full(len(D), np.nan)
    for k in range(nfold):
        te = np.flatnonzero(fold == k)
        if len(te):
            m = lgb.Booster(model_file=str(Path(model_dir) / f"{tag}_fold{k}.txt"))
            p = m.predict(_matrix(D, feats, te))
            if neg_rate < 1:
                p = neg_rate * p / (neg_rate * p + (1 - p))
            pred[te] = p
    assert not np.isnan(pred).any()
    return pred


def auc_by_group(y, p, groups):
    from sklearn.metrics import roc_auc_score
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.any() and 0 < y[m].sum() < m.sum():
            out[f"auc_{g}"] = float(roc_auc_score(y[m], p[m]))
    return out
